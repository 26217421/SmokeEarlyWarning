package smoke.da.model

import breeze.linalg.{DenseVector, rand, where}
import breeze.numerics.{floor, pow}
import breeze.stats.distributions.Uniform
import smoke.da.da.{DragonflyAlgorithm, Parameters}
import smoke.da.da.Tools.V

/**
  * Created by wojciech on 22.04.17. 
  */
class DA(f: DenseVector[Double] => Double, val nAgents: Int, lb: DenseVector[Double], ub: DenseVector[Double], parameters: (Int, Int) => Parameters)
  extends DragonflyAlgorithm(nAgents) {

  override def func(x: V): Double = f(x)

  override def radius(i: Int, max: Int): V = {
    val bounds = ub - lb
    (bounds *:* 0.25) + (bounds *:* ((i.toDouble / max.toDouble) * 2.0))
  }
  private val diff = ub - lb
  def border(pos: V, velocity: V): (V, V) = {
    val overIndex = pos >:> ub
    val underIndex = pos <:< lb

    pos(where(overIndex)) := lb(overIndex)
    pos(where(underIndex)) := ub(underIndex)

    velocity(where(overIndex)) := diff(where(overIndex)) :*= rand()
    velocity(where(underIndex)) := diff(where(underIndex)) :*= rand()
    (pos, velocity)
  }

  override def params(i: Int, max: Int): Parameters = parameters(i, max)

  override def randomAgents(): Iterator[Point] = {
    val uniform = lb.data.zip(ub.data).map(t => new Uniform(t._1, t._2))
    def vector(): DenseVector[Double] = DenseVector.apply(uniform.map(_.get()))
    Iterator.continually(Point(vector(), vector()))
  }

}
