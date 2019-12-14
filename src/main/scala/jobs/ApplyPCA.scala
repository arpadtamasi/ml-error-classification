package jobs

import org.apache.spark.ml.clustering.{GaussianMixture, KMeans, LDA}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.ml.{Model, Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object ApplyPCA  {

  private implicit val spark: SparkSession = {
    SparkSession
      .builder()
      .appName(getClass.getName)
      .master("local[*]")
      .getOrCreate()
  }

  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val path = args(0)
    val frameworkName = args(1)
    val className = args(2)
    val featureBits = args(3).toInt
    val dimensions = args(4).split(" ").map {_.toInt}
    val clusters = args(5).split(" ").map {_.toInt}
    val outPath = args(6)
    val urlCompatibleClassName = className.replace("::", "_")

    val hashSize = (1 << featureBits) - 1

    val rawDataset = spark.read.parquet(s"$path")
      .filter($"framework_name" === frameworkName && $"class_name" === className && $"message".isNotNull)

    val dataset = if (rawDataset.rdd.partitions.length < 60) rawDataset.repartition(60) else rawDataset

    //val model: PipelineModel = trainLSH(hashSize, dataset)

    val model: Model[_] with MLWritable = calibratePCA(hashSize, dimensions, clusters, dataset)

    val transformed = model.transform(dataset)

    transformed.write.mode(SaveMode.Overwrite).parquet(s"$outPath/$frameworkName/$urlCompatibleClassName/dataset")

    model.write.overwrite().save(s"$outPath/$frameworkName/$urlCompatibleClassName/model")
  }

  private def calibratePCA(hashSize: Int, pcaDims: Seq[Int], clusterKs: Seq[Int], dataset: DataFrame): Model[_] with MLWritable = {

    val pcaCal = pca(3, normalizedFeaturesCol, principalComponentsCol)
    val kmeansCal = kmeans(3, principalComponentsCol, clusterCol)

    val pipeline = new Pipeline().setStages(
      Array(
        tokenizer(messageCol, tokensCol),
        contextualizer,
        hasher(hashSize, contextualizedTokensCol, featuresCol),
        normalizer(featuresCol, normalizedFeaturesCol),
        pcaCal,
        kmeansCal
      ))

    if (pcaDims.size > 1 && clusterKs.size > 1) {
      val paramGrid = new ParamGridBuilder()
        .addGrid(pcaCal.k, pcaDims)
        .addGrid(kmeansCal.k, clusterKs)
        .build()

      val evaluator = new ClusteringEvaluator()
        .setFeaturesCol(featuresCol)
        .setPredictionCol(clusterCol)

      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2) // Use 3+ in practice
        .setParallelism(1) // Evaluate up to 2 parameter settings in parallel
      //.setCollectSubModels(true)

      cv.fit(dataset)
    } else {
      pcaCal.set(pcaCal.k, pcaDims.head)
      kmeansCal.set(kmeansCal.k, clusterKs.head)
      pipeline.fit(dataset)
    }
  }

  private def trainPCA(hashSize: Int, dimensions: Int, clusters: Int, dataset: DataFrame): PipelineModel = {
    val pipeline = new Pipeline().setStages(
      Array(
        tokenizer(messageCol, tokensCol),
        placeholderContextualizer,
        hasher(hashSize, contextualizedTokensCol, featuresCol),
        normalizer(featuresCol, normalizedFeaturesCol),
        pca(dimensions, normalizedFeaturesCol, principalComponentsCol),
        kmeans(clusters, principalComponentsCol, clusterCol)
      ))
    pipeline.fit(dataset)
  }

  private def trainLSH(hashSize: Int, dataset: DataFrame): PipelineModel = {
    val lshPipeline = new Pipeline().setStages(
      Array(
        tokenizer(messageCol, tokensCol),
        contextualizer,
        hasher(hashSize, contextualizedTokensCol, hashedTokensCol),
        lshMinHash(5, hashedTokensCol, lshHashes)
      ))

    val model = lshPipeline.fit(dataset)
    model
  }

  val messageCol = "message"
  val tokensCol = "tokens"
  val contextualizedTokensCol = "contextualized_tokens"
  val hashedTokensCol = "hashed_tokens"
  val normalizedFeaturesCol = "features_normalized"
  val principalComponentsCol = "principal_components"
  val clusterCol = "cluster"
  val lengthCol = "tokens_length"
  val messageLengthFeaturesCol = "length_feature"
  val featuresCol = "features"
  val probabilityCol = "probability"
  val topicDistributionCol = "topic_distribution"
  val lshHashes = "lsh_hashes"

  def tokenizer(inputCol: String, outputCol: String): Tokenizer = new Tokenizer()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)

  val contextualizer = new Contextualizer

  val placeholderContextualizer = new PlaceholderContextualizer()
  val templateContextualizer = new TemplateContextualizer()

  def hasher(hashSize: Int, inputCol: String, outputCol: String): HashingTF = new HashingTF()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .setNumFeatures(hashSize)
    .setBinary(true)

  def normalizer(inputCol: String, outputCol: String): StandardScaler = new StandardScaler()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .setWithMean(true)
    .setWithStd(true)

  def pca(dimensions: Int, inputCol: String, outputCol: String): PCA = new PCA()
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
    .setK(dimensions)

  def kmeans(clusters: Int, inputCol: String, outputCol: String): KMeans = new KMeans()
    .setFeaturesCol(inputCol)
    .setPredictionCol(outputCol)
    .setK(clusters)

  def gmm(clusters: Int, inputCol: String, outputCol: String, probabilityCol: String): GaussianMixture = new GaussianMixture()
    .setFeaturesCol(inputCol)
    .setPredictionCol(outputCol)
    .setProbabilityCol(probabilityCol)
    .setK(clusters)

  def lda(clusters: Int, inputCol: String, outputCol: String): LDA = new LDA()
    .setFeaturesCol(inputCol)
    .setTopicDistributionCol(outputCol)
    .setK(clusters)

  def lshMinHash(numHashTables: Int, inputCol: String, outputCol: String): MinHashLSH = new MinHashLSH()
    .setNumHashTables(numHashTables)
    .setInputCol(inputCol)
    .setOutputCol(outputCol)
}
