package jobs

import features.{RegexExtractor, SampleMessage}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.{SaveMode, SparkSession}

import scala.annotation.tailrec

object Agglomerative {
  private implicit val spark: SparkSession = {
    SparkSession
      .builder()
      .appName(getClass.getName)
      .master("local[*]")
      .getOrCreate()
  }

  private val tokenizer = new Tokenizer()
    .setInputCol("message")
    .setOutputCol("words")

  @tailrec
  def findPatterns(data: Seq[SampleMessage], maxGroups: Int, acc: Seq[(Option[String], Seq[SampleMessage])] = Nil): Seq[(Option[String], Seq[SampleMessage])] = {
    if (data.isEmpty) {
      acc
    } else {
      val (clusters: Seq[(String, Seq[SampleMessage])], singles: Seq[(String, Seq[SampleMessage])]) = RegexExtractor.collect(data, 1).partition {_._2.size > 1}
      val justFound = clusters map { case (r, messages) => Some(r) -> messages }
      if (clusters.isEmpty) {
        val noPattern = singles map { case (_, messages) => None -> messages }
        noPattern ++ justFound ++ acc
      } else {
        findPatterns(singles flatMap {_._2}, maxGroups + 1, acc)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    import spark.implicits._

    val path = args(0)
    val frameworkName = args(1)
    val className = args(2)
    val outPath = args(3)

    val rawDataset = spark.read.parquet(path)
      .filter($"framework_name" === frameworkName && $"class_name" === className && $"message".isNotNull)

    val dataset = if (rawDataset.rdd.partitions.length < 60) rawDataset.repartition(60) else rawDataset

    val tokenizer = new Tokenizer()
      .setInputCol("message")
      .setOutputCol("words")

    val tokenContext = udf { (frameworkName: String, className: String, tokens: Seq[String]) =>
      tokens.zipWithIndex map { case (t, i) =>
        s"$frameworkName|$className|$i|$t"
      }
    }

    val tokenized = tokenizer
      .transform(dataset)
      .withColumn("tokensWithContext", tokenContext(lit(frameworkName), lit(className), $"words"))

    val hashingTF = new HashingTF()
      .setInputCol("tokensWithContext")
      .setOutputCol("features")
      .setNumFeatures((1 << 18) - 1)
      .setBinary(true)

    val featured = hashingTF.transform(tokenized)

    val distinctFeatureVectors = featured.select($"features").distinct.collect.map {_.getAs[SparseVector](0)}
    val clusters = RegexExtractor.agglomerateClusters(distinctFeatureVectors, 3)
    val clusteredFeatures = clusters.zipWithIndex.flatMap { case (c, i) =>
      c.vectors.map { v => v -> i }
    }.toDF("features", "cluster")

    val clusteredDataset = featured
      .join(clusteredFeatures, Seq("features"))
      .select($"raw_item_id", $"message", $"cluster")

    val urlCompatibleClassName = className.replace("::", "_")
    clusteredDataset
      .write
      .mode(SaveMode.Overwrite)
      .parquet(s"$outPath/$frameworkName/$urlCompatibleClassName/clustered")

    val clustersDataset = RegexExtractor.generatePatterns(clusteredDataset.orderBy($"cluster"))

    clustersDataset
      .write
      .mode(SaveMode.Overwrite)
      .parquet(s"$outPath/$frameworkName/$urlCompatibleClassName/clusters")
  }
}
