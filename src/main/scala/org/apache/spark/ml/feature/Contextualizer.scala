package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse

class Contextualizer(val uid: String) extends Transformer with MLWritable {
  private val inputColName = "tokens"
  private val outputColName = "contextualized_tokens"
  private val lengthColName = "tokens_length"

  def this() = this(Identifiable.randomUID("contextualizer"))

  val tokenContext = udf { (frameworkName: String, className: String, tokens: Seq[String]) =>
    tokens.zipWithIndex map { case (t, i) =>
      s"$frameworkName|$className|$i|$t"
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    dataset
      .withColumn(
        outputColName, tokenContext(
          col("framework_name"),
          col("class_name"),
          col(inputColName)
        )
      )
      .withColumn(
        lengthColName, size(col(inputColName))
      )
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = {
    val outputDataType = new ArrayType(StringType, false)
    val inputFields = schema.fields
    require(
      inputFields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    val outputFields = inputFields :+ StructField(outputColName, outputDataType) :+ StructField(lengthColName, IntegerType, false)

    StructType(outputFields)
  }
  override def write: MLWriter = new ContextualizerWriter(this)

}


object Contextualizer extends MLReadable[Contextualizer] {
  override def read: MLReader[Contextualizer] = ContextualizerReader()
}

private class ContextualizerWriter(instance: Contextualizer) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
  }
}
case class ContextualizerReader() extends MLReader[Contextualizer] {
  override def load(path: String): Contextualizer = {
    implicit val format = DefaultFormats
    val rMetadataPath = new Path(path, "metadata").toString
    val modelPath = new Path(path, "model").toString

    val rMetadataStr = sc.textFile(rMetadataPath, 1).first()
    val rMetadata = parse(rMetadataStr)

    new Contextualizer()
  }
}
