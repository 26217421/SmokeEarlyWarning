package smoke {

  import breeze.linalg.DenseVector
  import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
  import org.apache.spark.ml.evaluation.Evaluator
  import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
  import org.apache.spark.ml.{Estimator, Model}
  import org.apache.spark.sql.{DataFrame, Dataset, Row}
  import smoke.Smoke.{processData, smote}
  import smoke.Train.kFold

  import scala.collection.mutable.ListBuffer

  class Train(var estimator: Estimator[_], var evaluator: Evaluator) {
    var n_tree: IntParam = estimator.asInstanceOf[RandomForestClassifier].numTrees
    var impurity: Param[String] = estimator.asInstanceOf[RandomForestClassifier].impurity
    val df: DataFrame = Smoke.getDF(1)
    val foldDf: Array[Dataset[Row]] = kFold(df)
    val paramMap: ParamMap = new ParamMap

    def train(foldDf: Array[DataFrame], paramMap: ParamMap, isSample:Boolean = false
             ): (Double, Model[_]) = {
      var metric: Double = 0
      val metrics = ListBuffer.empty[(Double, Model[_])]
      for ((k, _) <- foldDf.zipWithIndex) {
        var train_df = foldDf.filter(x => x != k).reduce(_ union _)
        if (isSample) {
          train_df = smote(train_df)
        } else {
          train_df = processData(train_df)
        }
        val val_df = processData(k)
        val model: Model[RandomForestClassificationModel] = estimator.fit(train_df, paramMap)
          .asInstanceOf[Model[RandomForestClassificationModel]]
        metric = evaluator.evaluate(model.transform(val_df, paramMap))
        metrics.append((metric, model))
      }
      metrics.maxBy(_._1)
    }
    def modelFitness(x: DenseVector[Double]): Double = {

      println(x.map(Math.round).toArray.mkString(" "))
       val str:String = if (math.round(x(1)).toInt != 0) {
        "entropy"
      } else {
        "gini"
      }
      paramMap put(n_tree, math.round(x(0)).toInt)
      paramMap put(impurity, str)
      val v: Double = train(foldDf, paramMap)._1
      println("F = {}", v)
      v
    }
  }

  object Train {
    def main(args: Array[String]): Unit = {

    }

    private def sub(a:Double, b:Double): Double = {
      val decimal1 = BigDecimal(a.toString)
      val decimal2 = decimal1.-(BigDecimal(b.toString))
      decimal2.toDouble
    }

    def kFold(df: DataFrame, k: Int = 5): Array[Dataset[Row]] = {
      val numFoldsF = k.toDouble
      val tmpA = (1 to k).map { fold =>
        fold / numFoldsF
      }.toArray
      val tmpB: Array[Double] = tmpA.sliding(2, 1).map(_.reduceLeft((a, b) => sub(b, a))).toArray
      val split = Array.concat(tmpA.take(1), tmpB)

      val foldDf: Array[Dataset[Row]] = df.randomSplit(split)
      foldDf
    }
  }
}
