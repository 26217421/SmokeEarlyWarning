package smoke

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.instance.ASMOTE
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql._

object Smoke extends SmokeStruct {
  val spark: SparkSession = SparkSession.builder()
    .appName("SparkSQLDemo")
    .master("local[*]")
    .getOrCreate()
  import spark.implicits._
  val fEvaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("fMeasureByLabel")
  val accEvaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
  //noinspection DuplicatedCode
  def main(args: Array[String]): Unit = {
    //run_v1(20)
    //run_v3(20, isSample = true)
    run_v1(112)
    //run_v3(26, isSample = true)

    spark.stop()
  }

  private def splitDataSet(): Unit = {
    val dataFrame = spark.read
      .format("csv")
      .option("header", "true")
      .schema(schema)
      .load("./input/smoke1.csv")
    val Array(trainData, testData) = dataFrame.randomSplit(Array(0.8, 0.2))
    trainData.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv("./input/train.csv")
    testData.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv("./input/test.csv")
  }

  def getDF: Array[Dataset[Row]] = {
    var trainData = spark.read
      .format("csv")
      .option("header", "true")
      .schema(schema)
      .load("./input/train.csv")
    var testData = spark.read
      .format("csv")
      .option("header", "true")
      .schema(schema)
      .load("./input/test.csv")
    trainData = processFeature(trainData)
    testData = processFeature(testData)
    trainData.cache()
    testData.cache()
    Array(trainData, testData)
  }

  def processFeature(df: DataFrame): DataFrame = {
    var dataFrame = df.withColumnRenamed("Fire Alarm", "label")
      .withColumn("label", $"label".cast(DataTypes.DoubleType))
    //dataFrame = dataFrame.drop("index", "UTC", "Raw Ethanol", "Humidity[%]", "CNT")
    dataFrame = dataFrame.drop("index", "CNT", "UTC")
    dataFrame
  }

  def run_v2():Unit = {
    var Array(trainData, testData) = getDF

    trainData = smote(trainData)
    trainData.groupBy("label").count().show()
    val rf: RandomForestClassifier = getSampleEstimator

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precisionByLabel")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val bestModel: RandomForestClassificationModel = selectBestModel(trainData, rf, evaluator, rf, sample = true)
    evaluate(testData, bestModel)
  }
  val train: Train = new Train(getSampleEstimator, fEvaluator)

  def run_v3(n:Int, isSample:Boolean): Unit = {
    val Array(_, testData) = getDF
    train.paramMap put(train.n_tree, n)
    train.paramMap put(train.impurity, "gini")
    println(n, isSample)
    val model = train.train(train.foldDf, train.paramMap, isSample)._2.asInstanceOf[RandomForestClassificationModel]
    evaluate(testData, model)
    model.write.overwrite().save(s"model/rf_$n")
  }

  def run_v1(n: Int): Unit = {
    val Array(trainData, testData) = getDF

    val inputCols = trainData.columns.filter(_ != "label")
    val (rf: RandomForestClassifier, pipeline: Pipeline) = getNoSamplePipline(inputCols)

    var data_10 = spark.read
      .format("csv")
      .option("header", "true")
      .schema(schema)
      .load("./input/data_10.csv")
    val model = pipeline.fit(trainData)
    val start = System.currentTimeMillis()
    val predictions = model.transform(data_10)
    println(System.currentTimeMillis() - start)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val bestModel: RandomForestClassificationModel = selectBestModel(trainData, pipeline, evaluator,
      rf, sample = false, n = n)
    evaluate(testData, bestModel)
  }

  def getSampleEstimator: RandomForestClassifier = {

    val rf = new RandomForestClassifier()
      .setNumTrees(112)
      .setLabelCol("label")
      .setFeaturesCol("features")
    rf
  }

  private def getNoSamplePipline(inputCols: Array[String]): (RandomForestClassifier, Pipeline) = {
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("features")
    val rf = new RandomForestClassifier()
      .setNumTrees(20)
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, rf))
    (rf, pipeline)
  }

  protected def selectBestModel(trainData: Dataset[Row], estimator: Estimator[_], evaluator:
  MulticlassClassificationEvaluator, rf: RandomForestClassifier, sample: Boolean, n: Int=20):
  RandomForestClassificationModel = {
    val paramGrid = new ParamGridBuilder()
      //.addGrid(rf.maxDepth, Array(10))
      .addGrid(rf.numTrees, Array(n))
      .addGrid(rf.impurity, Array("gini"))
      .build()
    val cv = new CrossValidator() //??????????????????????????????
      .setEstimator(estimator)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = cv.fit(trainData)
    println(cvModel.avgMetrics.mkString("Array(", ", ", ")"))
    if(!sample) {
      val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2)
        .asInstanceOf[RandomForestClassificationModel]
      println("The params of best RandomForestClassification model : " +
        bestModel.extractParamMap())
      bestModel
    } else {
      val bestModel = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]
      println("The params of best RandomForestClassification model : " +
        bestModel.extractParamMap())
      bestModel
    }

  }

  def evaluate(df: DataFrame, bestModel: RandomForestClassificationModel): Unit = {
    val res = bestModel.transform(processData(df))
    val accuracy = accEvaluator.evaluate(res)
    val scoreAndLabels: RDD[(Double, Double)] = res.select(bestModel.getProbabilityCol, bestModel.getLabelCol).rdd
      .map { row => {
        val pro = row.getAs[DenseVector](0)
        val label = row.getAs[Double](1)
        val score = pro(1)
        (score, label)
      }
      }

    val scoreAndLabels2 = res.select(bestModel.getPredictionCol, bestModel.getLabelCol).rdd
      .map { case Row(predLabel: Double, label: Double) => (predLabel, label) }
    val metric = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = metric.areaUnderROC()
    val pr = metric.areaUnderPR()
    val metric2 = new BinaryClassificationMetrics(scoreAndLabels2)
    val precision = metric2.precisionByThreshold
    val recall = metric2.recallByThreshold()
    val f1score = metric2.fMeasureByThreshold()
    val pr2 = metric2.areaUnderPR()
    val prc = metric2.pr()
    println(s"accuracy = $accuracy")
    println("auc:  " + auc)
    println("pr: " + pr)
    println("presion: " + precision.toJavaRDD().collect())
    println("recall: " + recall.toJavaRDD().collect())
    println("f1score: " + f1score.toJavaRDD().collect())
    println("pr2: " + pr2)
    println("prc:  " + prc.toJavaRDD().collect())
  }

  def processData(df_data: DataFrame): DataFrame = {

    val input = df_data.columns.clone().filter(_ != "label")

    val assembler = new VectorAssembler()
      .setInputCols(input)
      .setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("features")

    scaler.fit(assembler.transform(df_data)).transform(assembler.transform(df_data))
  }
  def smote(df_data: DataFrame): DataFrame = {

    val df = df_data
    val asmote = new ASMOTE().setK(5)
      .setPercOver(140)
      .setSeed(46)
    // oversampled DataFrame

    val newDF = asmote.transform(processData(df).select("features", "label"))
    //processData(df)
    newDF
  }
}
