package smoke.da.model

import breeze.linalg.Matrix.castUpdateOps
import breeze.linalg.{DenseVector, InjectNumericOps, rand, where}
import org.junit.Test


class DATest {
  val lb: DenseVector[Double] = DenseVector[Double](10, 0)
  val ub: DenseVector[Double] = DenseVector[Double](150, 1)
  val diff: DenseVector[Double] = ub - lb
  @Test
  def testBorder(): Unit = {
    val pos = DenseVector[Double](152.1, -1.992)
    val velocity = DenseVector[Double](52.002, 0.5)
    val overIndex = pos >:> ub
    val underIndex = pos <:< lb

    pos(where(overIndex)) := lb(overIndex)
    pos(where(underIndex)) := ub(underIndex)
    pos.foreach(println)
    //val f: DenseVector[Double] = floor(pos / diff +:+ 0.5)
    velocity(where(overIndex)) := diff(where(overIndex)) :*= rand()
    velocity(where(underIndex)) := diff(where(underIndex)) :*= rand()

    velocity.foreach(println)

  }
}
