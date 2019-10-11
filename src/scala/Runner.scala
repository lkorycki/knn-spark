import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Encoder, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object Runner {

  def main(args: Array[String]) {

    if (args.length < 3){
      println("Usage: $SPARK_HOME/bin/spark-submit --class \"Runner\" --master local[*] knn-spark.jar " +
        "dataset/file.arff K #partitions")
      System.exit(1)
    }

    val arffPath = args(0)
    val K = args(1).toInt
    val partitions = args(2).toInt
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("kNN")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val startTime = System.nanoTime

    val inputRows = spark.sparkContext
      .textFile(arffPath)
      .repartition(partitions)
      .filter((line: String) => !line.startsWith("@"))
      .map(_.split(",").to[List])
      .zipWithUniqueId()
      .map((columnsWithId) => toInstance(columnsWithId._1, columnsWithId._2))

    println("#Partitions: " + inputRows.getNumPartitions)

    val modelDS = spark.createDataset(inputRows)(instanceEncoder)
    //modelDS.printSchema()
    //modelDS.show()
    //modelDS.cache()

    val modelData = modelDS.collect()
    val testData = spark.sparkContext.broadcast(modelData)
    val k = spark.sparkContext.broadcast(K)

    val result = modelDS
      .mapPartitions((modelDataIt: Iterator[Instance]) => findPartitionNearestNeighbors(modelDataIt, testData.value, k.value))(nearestNeighborsEncoder)
      .groupByKey((nn: NearestNeighbors) => nn.instanceId)
      .reduceGroups((nn1: NearestNeighbors, nn2: NearestNeighbors) => reducePartitionsNearestNeighbors(nn1, nn2, k.value))
      .map(nn => classify(nn._2))
      .reduce(_ + _)

    val duration = (System.nanoTime - startTime) / 1e6d
    val accuracy = result / modelData.length.toDouble

    println(s"Accuracy: $accuracy\nTime: $duration ms")
    spark.stop()

  }

  def toInstance(rawColumns: List[String], id: Long): Instance = Instance(
    id,
    Vectors.dense(rawColumns.dropRight(1).map(l => l.toDouble).toArray),
    rawColumns.last.trim.toInt
  )

  def findPartitionNearestNeighbors(modelDataIt: Iterator[Instance], testData: Array[Instance], k: Int): Iterator[NearestNeighbors] = {
    val partitionNeighbors: ArrayBuffer[NearestNeighbors] = ArrayBuffer()
    val instancesPartitionNeighbors: ArrayBuffer[ArrayBuffer[ClassDist]] = ArrayBuffer.fill(
      testData.length)(ArrayBuffer.fill(k)(ClassDist(-1, Double.MaxValue)))

    while (modelDataIt.hasNext) {
      val modelInstance = modelDataIt.next()
      for ((testInstance: Instance, i: Int) <- testData.zipWithIndex) {
        if (testInstance.id != modelInstance.id) {
          val dist = Vectors.sqdist(testInstance.features, modelInstance.features)
          tryInsertNeighbor(instancesPartitionNeighbors(i), modelInstance.label, dist, k)
        }
      }
    }

    for ((testInstance: Instance, i: Int) <- testData.zipWithIndex) {
      partitionNeighbors += NearestNeighbors(
        testInstance.id,
        instancesPartitionNeighbors(i),
        testInstance.label
      )
    }

    partitionNeighbors.iterator
  }

  def tryInsertNeighbor(currentNeighbors: ArrayBuffer[ClassDist], label: Int, dist: Double, k: Int): Unit = {
    var i = 0
    var continue = true

    while (continue) {
      if (dist < currentNeighbors(i).dist) {

        var v = k - 1
        while (v > i) {
          currentNeighbors(v) = ClassDist(currentNeighbors(v - 1).label, currentNeighbors(v - 1).dist)
          v = v - 1
        }

        currentNeighbors(i) = ClassDist(label, dist)
        continue = false
      }

      i = i + 1
      if (i == k) continue = false
    }
  }

  def reducePartitionsNearestNeighbors(nn1: NearestNeighbors, nn2: NearestNeighbors, k: Int): NearestNeighbors = {
    val mergedNearestNeighbors: ArrayBuffer[ClassDist] = (nn1.neighbors ++ nn2.neighbors)
      .sortWith((cd1: ClassDist, cd2: ClassDist) => cd1.dist < cd2.dist).take(k)

    NearestNeighbors(nn1.instanceId, mergedNearestNeighbors, nn1.trueLabel)
  }

  def classify(instanceNearestNeighbors: NearestNeighbors): Int = {
    val votes: mutable.Map[Int, Int] = mutable.Map[Int, Int]()
    var maxClass = (0, 0)

    for (vote <- instanceNearestNeighbors.neighbors) {
      if (!votes.contains(vote.label)) {
        votes.put(vote.label, 1)
      } else {
        votes(vote.label) = votes(vote.label) + 1
      }

      if (votes(vote.label) > maxClass._2) maxClass = (vote.label, votes(vote.label))
    }

    val prediction = maxClass._1
    if (prediction == instanceNearestNeighbors.trueLabel) 1 else 0
  }

  val instanceEncoder: Encoder[Instance] = ExpressionEncoder.apply[Instance]()
  val nearestNeighborsEncoder: Encoder[NearestNeighbors] = ExpressionEncoder.apply[NearestNeighbors]()

}

case class Instance(id: Long, features: Vector, label: Int)
case class NearestNeighbors(instanceId: Long, var neighbors: ArrayBuffer[ClassDist], trueLabel: Int)
case class ClassDist(label: Int, dist: Double)