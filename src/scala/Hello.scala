import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Encoder, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

object Hello {

  def main(args: Array[String]) {

    val instanceEncoder: Encoder[Instance] = ExpressionEncoder.apply[Instance]()
    val nearestNeighborsEncoder: Encoder[NearestNeighbors] = ExpressionEncoder.apply[NearestNeighbors]()

    val arffPath = "data\\small.arff"
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    //val log = Logger.getLogger(getClass.getName)

    val spark = SparkSession
      .builder()
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val inputRows = spark.sparkContext
      .textFile(arffPath)
      .repartition(8)
      .filter((line: String) => !line.startsWith("@"))
      .map(_.split(",").to[List])
      .map(toInstance)

    println(inputRows.getNumPartitions)

    val inputDS = spark.createDataset(inputRows)(instanceEncoder)
    inputDS.printSchema()
    inputDS.show()
    //inputDF.cache()

    val testData = spark.sparkContext.broadcast(inputDS.collect())

    val result = inputDS.mapPartitions((it: Iterator[Instance]) => {
      println("dupa " + testData)

      val nearestsNeighbors = Array(NearestNeighbors(0, Array()))

      nearestsNeighbors.iterator
    })(nearestNeighborsEncoder)
      .groupByKey((nn: NearestNeighbors) => nn.instanceIdx)
      .reduceGroups((nn1: NearestNeighbors, nn2: NearestNeighbors) => nn1)
      .map(t => 1)
      .reduce((a,b) => 0) // incremental average


    spark.stop()

  }

  def toInstance(rawColumns: List[String]): Instance = Instance(
    Vectors.dense(rawColumns.dropRight(1).map(l => l.toDouble).toArray),
    rawColumns.last.toInt
  )

}

case class Instance(features: Vector, label: Int)
case class NearestNeighbors(instanceIdx: Int, neighbors: Array[(Int, Double)])