# Distributed k-NN
k-NN algorithm implemented using Apache Spark

### Install
* Scala 2.11.11
* Spark 2.2.0
* sbt
* Dependencies given in *build.sbt*

### Run
```
$SPARK_HOME/bin/spark-submit --class "Runner" --master local[*] knn-spark.jar dataset/file.arff K #partitions
```
