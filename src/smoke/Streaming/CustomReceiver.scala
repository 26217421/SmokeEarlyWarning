package smoke.Streaming

import jline.internal.InputStreamReader
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.receiver.Receiver

import java.io.BufferedReader
import java.net.Socket
import java.nio.charset.StandardCharsets

class CustomReceiver(host: String, port: Int) extends Receiver[String](StorageLevel.MEMORY_ONLY){
  override def onStart(): Unit = {
    new Thread("Socket Receiver") {
      override def run(): Unit = {
        receive()
      }
    }
  }

  def receive(): Unit = {
    val socket: Socket = new Socket(host, port)
    var input: String = null
    val reader = new BufferedReader(new InputStreamReader(socket.getInputStream, StandardCharsets.UTF_8))
    input = reader.readLine()
    while(!isStopped() && input != null) {
      store(input)
      input = reader.readLine()
    }
    reader.close()
    socket.close()
    restart("restart")
  }
  override def onStop(): Unit = {}
}
