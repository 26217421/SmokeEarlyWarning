package smoke

import breeze.linalg.DenseVector
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import smoke.da.model.{DA, VariableParam}

import java.io.{File, PrintWriter}

object DAOmp {
  val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("fMeasureByLabel")
  var opt = new Train(Smoke.getSampleEstimator, evaluator)
  def main(args: Array[String]): Unit = {
    omp()
  }
  def omp(): DA#Result = {
    val nAgents = 10
    val iterations = 10
    def parameters(iteration: Int, maxIteration: Int) =
      VariableParam(iteration, maxIteration)


    val lb: DenseVector[Double] = DenseVector[Double](10, 0)
    val ub: DenseVector[Double] = DenseVector[Double](150, 0)
    val fit = opt.modelFitness _
    val results = (new DA(fit, nAgents, lb, ub, parameters).iterator(iterations)
      take iterations).toList
    val jsonMapper = new ObjectMapper()
    jsonMapper.registerModule(DefaultScalaModule)
    val res: String = jsonMapper.writeValueAsString(results)

    val writer = new PrintWriter(new File("f_res.json"))
    writer.write(res)
    writer.close()
    val result = results.last

    result
  }
}
