package smoke

import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

trait SmokeStruct {
  val schema: StructType = StructType(List(
    StructField("index", DataTypes.LongType),
    StructField("UTC", DataTypes.LongType),
    StructField("Temperature[C]", DataTypes.DoubleType),
    StructField("Humidity[%]", DataTypes.DoubleType),
    StructField("TVOC[ppb]", DataTypes.LongType),
    StructField("eCO2[ppm]", DataTypes.LongType),
    StructField("Raw H2", DataTypes.LongType),
    StructField("Raw Ethanol", DataTypes.LongType),
    StructField("Pressure[hPa]", DataTypes.DoubleType),
    StructField("PM1", DataTypes.DoubleType),
    StructField("PM2_5", DataTypes.DoubleType),
    StructField("NC0_5", DataTypes.DoubleType),
    StructField("NC1_0", DataTypes.DoubleType),
    StructField("NC2_5", DataTypes.DoubleType),
    StructField("CNT", DataTypes.IntegerType),
    StructField("Fire Alarm", DataTypes.IntegerType)
  ))
}
