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
  var train = new Train(Smoke.getSampleEstimator, evaluator)
  def main(args: Array[String]): Unit = {
    val result = omp()
    val jsonMapper = new ObjectMapper()
    jsonMapper.registerModule(DefaultScalaModule)
    val v: String = jsonMapper.writeValueAsString(result)
    val writer = new PrintWriter(new File("res.json"))
    writer.write(v)
    writer.close()

    println(v)
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

    result
  }
}
