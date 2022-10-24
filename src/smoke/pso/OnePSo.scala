package smoke.pso



import scala.util.Random

object OnePSo extends Serializable {
  def apply() = {

  }
  var M: Int = 200; // 迭代次数
  var numParticles = 50; //粒子数量
  var dimension: Int = 3; //粒子的维数
  val pBest: Array[Array[Double]] = Array.ofDim[Double](numParticles, dimension)
  val xPos: Array[Array[Double]] = Array.ofDim[Double](numParticles, dimension)
  val xVel: Array[Array[Double]] = Array.ofDim[Double](numParticles, dimension)

  var gBest = new Array[Double](dimension)
  var fitness = new Array[Double](numParticles)

  var w = 0.5
  var c1 = 2.0
  var c2 = 2.0
  var xMax = 10
  var xMin = 0
  var vMax = 5
  var vMin = -5


  def main(args: Array[String]): Unit = {
    Search()
  }

  def calaFitness(temp: Array[Double]): Double = {
    var y: Double = 0.0
    for (i <- 0 to dimension-1) {
      y += temp(i) * temp(i)
    }
    return y
  }


  def Initializes(): Unit = {
    //初始化位置和速度
    for (i <- 0 until numParticles) {
      for (j <- 0 until dimension) {
        xPos(i)(j) = xMin + Random.nextDouble() * (xMax - xMin)
        xVel(i)(j) = vMin + Random.nextDouble() * (vMax - vMin)
      }
    }
    //计算每个粒子的适应值，并初始化局部和全局最优解
    for (i <- 0 to numParticles-1) {
      fitness(i) = calaFitness(xPos(i))
      for (j <- 0 to dimension-1) {
        pBest(i)(j) = xPos(i)(j)
      }
    }
    // 初始化最优适应值对应的位置gBest
    var bestFitness = fitness(0)
    var isChange = false
    for (i <- 1 to numParticles-1) {
      if (fitness(i) < bestFitness) {
        isChange = true
        bestFitness = fitness(i)
        for (j <- 0 to dimension-1) {
          gBest(j) = xPos(i)(j)
        }
      }
    }
    if (isChange == false) {
      for (i <- 0 to dimension-1) {
        gBest(i) = xPos(0)(i)
      }
    }
    //输出初始化提示信息
    println("初始化完毕！")
    print("0 ----> " + calaFitness(gBest) + " ----> [")
    for (i <- 0 to dimension-1) {
      print(gBest(i) + ",\t")
    }
    println("]")
  }


  def Search(): Unit = {
    Initializes()
    for (m <- 0 to M-1) {
      for (n <- 0 to numParticles-1) {
        //更新计算出粒子的速度和位置
        for (d <- 0 to dimension-1) {
          xVel(n)(d) = w * xVel(n)(d) + c1 * Random.nextDouble() * (pBest(n)(d)
            - xPos(n)(d)) + c2 * Random.nextDouble() * (gBest(d) - xPos(n)(d))
          xPos(n)(d) = xPos(n)(d) + xVel(n)(d)
        }
        //将粒子新的位置对应的适应值与原先适应值相比较，如果新位置的适应值较小，则更新此粒子历史最优位置
        if (calaFitness(xPos(n)) < fitness(n)) {
          fitness(n) = calaFitness(xPos(n))
          for (d <- 0 to dimension-1) {
            pBest(n)(d) = xPos(n)(d)
          }
        }
        // 判断是否需要更新全局最优位置
        if (fitness(n) < calaFitness(gBest)) {
          for (d <- 0 until dimension) {
            gBest(d) = pBest(n)(d)
          }
        }
        // 输出这一迭代步骤完成后全局的最优值以及相对应的位置信息
        print((m + 1) + " ----> " + calaFitness(gBest) + " ----> [")
        for (i <- 0 until dimension) {
          print(gBest(i) + ", \t")
        }
        println("]")
      }

    }
  }
}

class OnePSo (f:(Int,Int)=>Double, val dimension:Int) {

  }
