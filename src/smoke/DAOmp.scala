package smoke

import breeze.linalg.DenseVector
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import smoke.da.model.{DA, VariableParam}

object DAOmp {
  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
  var train = new Train(Smoke.getSampleEstimator, evaluator)
  def main(args: Array[String]): Unit = {
    var result = omp()

  }
  def omp(): DA#Result = {
    val nAgents = 20
    val iterations = 50
    def parameters(iteration: Int, maxIteration: Int) =
      VariableParam(iteration, maxIteration)


    val lb: DenseVector[Double] = DenseVector[Double](10, 0)
    val ub: DenseVector[Double] = DenseVector[Double](150, 1)
    val fit = train.modelFitness _
    val result = (new DA(fit, nAgents, lb, ub, parameters).iterator(iterations)
      take iterations).toList.last

    val value = result.global.value
    result
  }
}
