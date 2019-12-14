package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.DefaultFormats

class TemplateContextualizer(val uid: String) extends Transformer with MLWritable {
  private val inputColName = "tokens"
  private val outputColName = "contextualized_tokens"
  private val lengthColName = "tokens_length"

  def this() = this(Identifiable.randomUID("template_contextualizer"))

  val tokenContext: UserDefinedFunction = udf { (frameworkName: String, className: String, tokens: Seq[String]) =>
    tokens.indices.map { i =>
      val (l, r) = tokens.splitAt(i)
      l ++ r.tail
    }.map {_.mkString("||")} map { template =>
      s"$frameworkName|$className|$template"
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
  override def write: MLWriter = new TemplateContextualizerWriter(this)
}

object TemplateContextualizer extends MLReadable[TemplateContextualizer] {
  override def read: MLReader[TemplateContextualizer] = new TemplateContextualizerReader()
}

private class TemplateContextualizerWriter(instance: TemplateContextualizer) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
  }
}
case class TemplateContextualizerReader() extends MLReader[TemplateContextualizer] {
  override def load(path: String): TemplateContextualizer = {
    implicit val format: DefaultFormats.type = DefaultFormats
    val rMetadataPath = new Path(path, "metadata").toString

    val rMetadataStr = sc.textFile(rMetadataPath, 1).first()

    //    val modelPath = new Path(path, "model").toString
    //    val rMetadata = parse(rMetadataStr)

    new TemplateContextualizer()
  }
}
