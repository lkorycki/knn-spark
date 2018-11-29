import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Encoder, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

import scala.collection.mutable
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
    //modelDS.cache()

    val modelData = modelDS.collect()
    val testData = spark.sparkContext.broadcast(modelData)
    val k = spark.sparkContext.broadcast(K)

    val result = modelDS
      .mapPartitions((it: Iterator[Instance]) => findPartitionNearestNeighbors(it, testData.value, k.value))(nearestNeighborsEncoder)
      .groupByKey((nn: NearestNeighbors) => nn.instanceIdx)
      .reduceGroups((nn1: NearestNeighbors, nn2: NearestNeighbors) => reducePartitionNearestNeighbors(nn1, nn2, k.value))
      .map((instNN: (Int, NearestNeighbors)) => classify(instNN._1, instNN._2, testData.value))
      .reduce(_ + _)

    val duration = (System.nanoTime - startTime) / 1e9d
    val accuracy = result / modelData.length.toDouble

    println(s"Accuracy: $accuracy\nTime: $duration s")
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
      partitionNeighbors += NearestNeighbors(i, instancesPartitionNeighbors(i).sortBy((cd: ClassDist) => cd.dist).take(k))
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

  def classify(instanceId: Int, instanceNearestNeighbors: NearestNeighbors, testData: Array[Instance]): Int = {
    val votes: mutable.Map[Int, Int] = mutable.Map[Int, Int]()

    for (vote <- instanceNearestNeighbors.neighbors) {
      if (!votes.contains(vote.label)) {
        votes.put(vote.label, 1)
      } else {
        votes.put(votes(vote.label), votes(vote.label) + 1)
      }
    }

    val prediction = votes.maxBy(_._2)._1
    println(prediction, testData(instanceId).label)
    if (prediction == testData(instanceId).label) 1 else 0
  }

  val instanceEncoder: Encoder[Instance] = ExpressionEncoder.apply[Instance]()
  val nearestNeighborsEncoder: Encoder[NearestNeighbors] = ExpressionEncoder.apply[NearestNeighbors]()

}

case class Instance(features: Vector, label: Int)
case class NearestNeighbors(instanceIdx: Int, var neighbors: ArrayBuffer[ClassDist])
case class ClassDist(label: Int, dist: Double)