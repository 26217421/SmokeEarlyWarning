package smoke.Streaming

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import smoke.Smoke.{evaluate, getDF}

object WarningApp extends App {
  val conf = new SparkConf().setMaster("local[*]").setAppName("StreamWordCount")
  val ssc = new StreamingContext(conf, Seconds(3))
//  val rf = RandomForestClassificationModel.load("./model/rf_112")
//  println(rf.extractParamMap())
  val lineStreams = ssc.socketTextStream("localhost", 11223)
  lineStreams.start()

  lineStreams.print()
  ssc.start()
  ssc.awaitTermination()
}
