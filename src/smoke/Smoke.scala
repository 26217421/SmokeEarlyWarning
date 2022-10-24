package smoke

import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler, VectorAssembler}
import org.apache.spark.ml.instance.ASMOTE
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Smoke extends SmokeStruct {
  val spark: SparkSession = SparkSession.builder()
    .appName("SparkSQLDemo")
    .master("local[*]")
    .getOrCreate()
  import spark.implicits._

  //noinspection DuplicatedCode
  def main(args: Array[String]): Unit = {
    val dataFrame = spark.read
      .format("csv")
      .option("header", "true")
      .schema(schema)
      .load("./input/smoke1.csv")
    //run_v1(dataFrame)
    run_v1(dataFrame)
    spark.stop()
  }
  def run_v2(ds: Dataset[Row]):Unit = {
    var dataFrame = ds.toDF()
    dataFrame = dataFrame.withColumnRenamed("Fire Alarm", "label")
      .withColumn("label", $"label".cast(DataTypes.DoubleType))
    dataFrame.drop("index", "UTC")
    var Array(trainData, testData) = dataFrame.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()
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


  def run_v1(ds: Dataset[Row]): Unit = {
    var dataFrame = ds.toDF()
    //dataFrame.show()
    dataFrame.groupBy("Fire Alarm").count().show()
    dataFrame.drop("index", "UTC", "Raw Ethanol", "Humidity[%]")

    dataFrame = dataFrame.withColumnRenamed("Fire Alarm", "label")
      .withColumn("label", $"label".cast(DataTypes.DoubleType))

    val Array(trainData, testData) = dataFrame.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    testData.cache()
    testData.groupBy("label").count().show()
    val inputCols = trainData.columns.filter(_ != "label")
    val (rf: RandomForestClassifier, pipeline: Pipeline) = getNoSamplePipline(inputCols)
    val model = pipeline.fit(trainData)

    val predictions = model.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy = {}", accuracy)

    val bestModel: RandomForestClassificationModel = selectBestModel(trainData, pipeline, evaluator, rf, sample = false)
    evaluate(testData, bestModel)
  }

  private def getSampleEstimator: RandomForestClassifier = {

    val rf = new RandomForestClassifier()
      .setMaxDepth(10)
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
      .setNumTrees(100)
      .setMaxDepth(10)
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, rf))
    (rf, pipeline)
  }

  protected def selectBestModel(trainData: Dataset[Row], estimator: Estimator[_], evaluator: MulticlassClassificationEvaluator, rf: RandomForestClassifier, sample: Boolean): RandomForestClassificationModel = {
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(10))
      //.addGrid(rf.numTrees, Range(100, 150, 10))
      .addGrid(rf.impurity, Array("entropy"))
      .build()
    val cv = new CrossValidator() //使用交叉验证训练模型
      .setEstimator(estimator)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

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

  protected def evaluate(df: DataFrame, bestModel: RandomForestClassificationModel): Unit = {
    val res = bestModel.transform(processData(df))
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

    println("\n auc:  " + auc)
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

    val scaler = new MinMaxScaler()
      .setInputCol("featureVector")
      .setOutputCol("features")

    scaler.fit(assembler.transform(df_data)).transform(assembler.transform(df_data))
  }
  def smote(df_data: DataFrame): DataFrame = {
    import spark.implicits._

    val df = df_data.withColumnRenamed("Fire Alarm", "label")
      .withColumn("label", $"label".cast(DataTypes.DoubleType))
    val asmote = new ASMOTE().setK(5)
      .setPercOver(145)
      .setSeed(46)
    // oversampled DataFrame

    val newDF = asmote.transform(processData(df).select("features", "label"))
    newDF
  }
}
