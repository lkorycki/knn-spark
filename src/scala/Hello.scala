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
      .repartition(1)
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
      .mapPartitions((it: Iterator[Instance]) => findPartitionNearestNeighbors(it, testData.value, k.value))(nearestNeighborsEncoder)
      .groupByKey((nn: NearestNeighbors) => nn.instanceIdx)
      .reduceGroups((nn1: NearestNeighbors, nn2: NearestNeighbors) => reducePartitionNearestNeighbors(nn1, nn2, k.value))

    val result = partitionsNeighbors
      .map(t => {
        println(t)
        1
      })
      .reduce((a,b) => 0) // incremental average

    val duration = (System.nanoTime - startTime) / 1e9d

    println(s"Accuracy: $result\nTime: $duration s")
    spark.stop()

  }

  def toInstance(rawColumns: List[String]): Instance = Instance(
    Vectors.dense(rawColumns.dropRight(1).map(l => l.toDouble).toArray),
    rawColumns.last.toInt
  )

  def findPartitionNearestNeighbors(it: Iterator[Instance], testData: Array[Instance], k: Int): Iterator[NearestNeighbors] = {
    val partitionNeighbors: ArrayBuffer[NearestNeighbors] = ArrayBuffer()
    val instancesPartitionNeighbors: ArrayBuffer[ArrayBuffer[ClassDist]] = ArrayBuffer.fill(testData.length)(ArrayBuffer())

    while (it.hasNext) {
      val modelInstance = it.next()

      for ((testInstance: Instance, i: Int) <- testData.zipWithIndex) {
          val dist = Vectors.sqdist(testInstance.features, modelInstance.features)
          instancesPartitionNeighbors(i) += ClassDist(modelInstance.label, dist)
      }
    }

    for ((testInstance: Instance, i: Int) <- testData.zipWithIndex) {
      instancesPartitionNeighbors(i).sortBy((cd: ClassDist) => cd.dist)
      partitionNeighbors += NearestNeighbors(i, instancesPartitionNeighbors(i).take(k))
    }

    partitionNeighbors.iterator
  }

  def reducePartitionNearestNeighbors(nn1: NearestNeighbors, nn2: NearestNeighbors, k: Int): NearestNeighbors = {
    val mergedNearestNeighbors: ArrayBuffer[ClassDist] = ArrayBuffer.fill(k)(null)

    (0 until k).foreach(i => {
      mergedNearestNeighbors(i) = if (nn1.neighbors(i).dist < nn2.neighbors(i).dist) nn1.neighbors(i) else nn2.neighbors(i)
    })

    NearestNeighbors(nn1.instanceIdx, mergedNearestNeighbors)
  }

  val instanceEncoder: Encoder[Instance] = ExpressionEncoder.apply[Instance]()
  val nearestNeighborsEncoder: Encoder[NearestNeighbors] = ExpressionEncoder.apply[NearestNeighbors]()

}

case class Instance(features: Vector, label: Int)
case class NearestNeighbors(instanceIdx: Int, var neighbors: ArrayBuffer[ClassDist])
case class ClassDist(label: Int, dist: Double)