import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Encoder, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import scala.collection.mutable.ArrayBuffer

object Hello {

  def main(args: Array[String]) {

    val arffPath = "data\\small.arff"
    val K = 3
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val startTime = System.nanoTime

    val inputRows = spark.sparkContext
      .textFile(arffPath)
      .repartition(8)
      .filter((line: String) => !line.startsWith("@"))
      .map(_.split(",").to[List])
      .map(toInstance)

    println(inputRows.getNumPartitions)

    val modelDS = spark.createDataset(inputRows)(instanceEncoder)
    modelDS.printSchema()
    modelDS.show()
    //inputDF.cache()

    val testData = spark.sparkContext.broadcast(modelDS.collect())
    val k = spark.sparkContext.broadcast(K)

    val partitionsNeighbors = modelDS
      .mapPartitions((it: Iterator[Instance]) => findParitionNearestNeighbors(it, testData.value, k.value))(nearestNeighborsEncoder)

    val result = partitionsNeighbors
      .groupByKey((nn: NearestNeighbors) => nn.instanceIdx)
      .reduceGroups((nn1: NearestNeighbors, nn2: NearestNeighbors) => nn1)
      .map(t => 1) // testData(idx) == pred
      .reduce((a,b) => 0) // incremental average

    val duration = (System.nanoTime - startTime) / 1e9d

    println(s"Accuracy: $result\nTime: $duration d")
    spark.stop()

  }

  def toInstance(rawColumns: List[String]): Instance = Instance(
    Vectors.dense(rawColumns.dropRight(1).map(l => l.toDouble).toArray),
    rawColumns.last.toInt
  )

  def findParitionNearestNeighbors(it: Iterator[Instance], testData: Array[Instance], k: Int): Iterator[NearestNeighbors] = {
    val partitionNeighbors: ArrayBuffer[NearestNeighbors] = ArrayBuffer()
    val testDataValues = testData

    for (testInstance: Instance <- testDataValues) {
      val instancePartitionNeighbors: ArrayBuffer[ClassDist] = ArrayBuffer()

      while (it.hasNext) {
        val modelInstance = it.next()
        val dist = Vectors.sqdist(testInstance.features, modelInstance.features)
        instancePartitionNeighbors += ClassDist(modelInstance.label, dist)
      }

      instancePartitionNeighbors.sortBy((cd: ClassDist) => cd.dist)
      partitionNeighbors += NearestNeighbors(testInstance.label, instancePartitionNeighbors.take(k))
    }

    partitionNeighbors.iterator
  }

  val instanceEncoder: Encoder[Instance] = ExpressionEncoder.apply[Instance]()
  val nearestNeighborsEncoder: Encoder[NearestNeighbors] = ExpressionEncoder.apply[NearestNeighbors]()

}

case class Instance(features: Vector, label: Int)
case class NearestNeighbors(instanceIdx: Int, var neighbors: ArrayBuffer[ClassDist])
case class ClassDist(label: Int, dist: Double) extends Ordered[ClassDist] {
  override def compare(that: ClassDist): Int = if (this.dist < that.dist) -1 else 1
}