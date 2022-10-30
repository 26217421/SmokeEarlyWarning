package smoke {

  import breeze.linalg.DenseVector
  import org.apache.spark.ml.classification.RandomForestClassifier
  import org.apache.spark.ml.evaluation.Evaluator
  import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
  import org.apache.spark.ml.{Estimator, Model}
  import org.apache.spark.sql.{DataFrame, Dataset, Row}
  import smoke.Smoke.{processData, smote}
  import smoke.Train.kFold

  class Train(var estimator: Estimator[_], var evaluator: Evaluator) {
    var n_tree: IntParam = estimator.asInstanceOf[RandomForestClassifier].numTrees
    var impurity: Param[String] = estimator.asInstanceOf[RandomForestClassifier].impurity


    def train(df: DataFrame, k: Int, paramMap: ParamMap): Double = {
      val split = kFold(k)
      val foldDf: Array[Dataset[Row]] = df.randomSplit(split)
      var metric: Double = 0
      for ((k, i) <- foldDf.zipWithIndex) {
        println(i)
        foldDf(i).groupBy("label").count().show()
        var train_df = foldDf.filter(x => x != k).reduce(_ union _)
        train_df = smote(train_df)
        val val_df = processData(k)
        val model: Model[_] = estimator.fit(train_df, paramMap).asInstanceOf[Model[_]]
        metric = evaluator.evaluate(model.transform(val_df, paramMap))
      }
      metric
    }
    def modelFitness(x: DenseVector[Double], df:DataFrame): Double = {
      val paramMap = new ParamMap
       val str:String = if (x(1).toInt != 0) {
        "entropy"
      } else {
        "gini"
      }
      paramMap put(n_tree, x(0).toInt)
      paramMap put(impurity, str)
      train(df, 5, paramMap)
    }
  }

  object Train {
    def main(args: Array[String]): Unit = {
      println(kFold(10).sum)

    }

    def kFold(k: Int): Array[Double] = {
      val numFoldsF = k.toDouble
      val tmpA = (1 to k).map { fold =>
        (fold / numFoldsF)
      }.toArray
      val tmpB: Array[Double] = tmpA.sliding(2, 1).map(_.reduceLeft((a, b) => b - a)).toArray
      println(tmpB.mkString)
      Array.concat(tmpA.take(1), tmpB)
    }


  }
}
