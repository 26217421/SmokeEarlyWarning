package smoke {

  import breeze.linalg.DenseVector
  import org.apache.spark.ml.classification.RandomForestClassifier
  import org.apache.spark.ml.evaluation.Evaluator
  import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
  import org.apache.spark.ml.{Estimator, Model}
  import org.apache.spark.sql.{DataFrame, Dataset, Row}
  import smoke.Smoke.processData
  import smoke.Train.kFold

  import scala.collection.mutable.ListBuffer

  class Train(var estimator: Estimator[_], var evaluator: Evaluator) {
    var n_tree: IntParam = estimator.asInstanceOf[RandomForestClassifier].numTrees
    var impurity: Param[String] = estimator.asInstanceOf[RandomForestClassifier].impurity
    val df: DataFrame = Smoke.getDF(1)
    val foldDf: Array[Dataset[Row]] = kFold(df)
    def train(foldDf: Array[DataFrame], paramMap: ParamMap): Double = {

      var metric: Double = 0
      val metrics = ListBuffer.empty[Double]
      for ((k, i) <- foldDf.zipWithIndex) {
        var train_df = foldDf.filter(x => x != k).reduce(_ union _)
        //train_df = smote(train_df)
        train_df = processData(train_df)
        val val_df = processData(k)
        val model: Model[_] = estimator.fit(train_df, paramMap).asInstanceOf[Model[_]]
        metric = evaluator.evaluate(model.transform(val_df, paramMap))
        metrics.append(metric)
      }

      metrics.max
    }
    def modelFitness(x: DenseVector[Double]): Double = {

      val paramMap = new ParamMap
      println(x.map(Math.round).toArray.mkString(" "))
       val str:String = if (math.round(x(1)).toInt != 0) {
        "entropy"
      } else {
        "gini"
      }
      paramMap put(n_tree, math.round(x(0)).toInt)
      paramMap put(impurity, str)
      val v: Double = train(foldDf, paramMap)
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
