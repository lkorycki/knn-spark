import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.io.Source

object Hello {

  def main(args: Array[String]) {

    val arffPath = "data\\small.arff"
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    //val log = Logger.getLogger(getClass.getName)

    def schema: StructType =
      StructType(
        Seq(
          StructField(name = "features", dataType = VectorType, nullable = false),
          StructField(name = "class", dataType = IntegerType, nullable = false)
        )
      )

    def toRow(rawColumns: List[String]): Row = Row(
      Vectors.dense(rawColumns.dropRight(1).map(l => l.toDouble).toArray),
      rawColumns.last.toInt
    )

    val spark = SparkSession
      .builder()
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()

    val inputRows = spark.sparkContext
      .textFile(arffPath)
      .filter((line: String) => !line.startsWith("@"))
      .map(_.split(",").to[List])
      .map(toRow)

    val inputDF = spark.createDataFrame(inputRows, schema)

    inputDF.printSchema()
    inputDF.show()

//    val numAs = logData.filter(line => line.contains("a")).count()
//    val numBs = logData.filter(line => line.contains("b")).count()
//    println(s"Lines with a: $numAs, Lines with b: $numBs")

    spark.stop()

  }

  def parseFile(path: String): Unit = {

    for (line <- Source.fromFile(path).getLines) {
      println(line)
    }
  }

}
