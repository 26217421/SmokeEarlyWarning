package smoke.Streaming

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import smoke.{Smoke, SmokeStruct}

object WarningApp extends App with SmokeStruct {
  val conf = new SparkConf().setMaster("local[*]").setAppName("StreamWordCount")
  val ssc = new StreamingContext(conf, Seconds(3))
  val spark: SparkSession = SparkSession.builder()
    .appName("SparkSQLDemo")
    .master("local[*]")
    .getOrCreate()
  val rf = RandomForestClassificationModel.load("./model/rf_112")
  import spark.implicits._

  val lineStreams = ssc.socketTextStream("106.54.171.108", 9000)
  lineStreams.foreachRDD(rdd => {
    val df =spark.read.json(rdd.toDS())
    df.show()
    if(! df.isEmpty)
      rf.transform(Smoke.processData(df)).show()

  })
  //lineStreams.print()
  ssc.start()
  ssc.awaitTermination()
}
