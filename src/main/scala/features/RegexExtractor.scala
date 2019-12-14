package features

import java.util.StringTokenizer

import org.apache.spark.ml.linalg._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.tailrec
import scala.collection.parallel.ParIterable
import scala.util.Try
import scala.util.control.NonFatal

object RegexExtractor  {

  private val Digits = """(\d+)"""
  private val Word = """(\w+)"""
  private val AnyString = """(\S+)"""

  def tokenize(s: String): Seq[String] = {
    tokenize(s, "\n\t() ")
  }

  def tokenize(s: String, delimiters: String): List[String] = {
    val tokenizer = new StringTokenizer(s, delimiters, true)
    lazy val iterator: Iterator[String] = new Iterator[String] {
      override def hasNext: Boolean = tokenizer.hasMoreTokens
      override def next(): String = tokenizer.nextToken()
    }

    iterator.toList
  }
  def tryExtractRegex(strings: Seq[String], maxGroups: Option[Int] = None): String = try {
    extractRegex(strings, maxGroups)
  } catch {
    case NonFatal(x) =>
      null
  }

  def generatePatterns(clusteredDataset: DataFrame)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    clusteredDataset
      .groupByKey(_.getAs[Int]("cluster"))
      .mapGroups { case (c, rows) =>
        val messages = rows.map {_.getAs[String]("message")}.toSeq
        val distinctMessages = messages.distinct
        val regex = RegexExtractor.tryExtractRegex(distinctMessages)
        (c, distinctMessages.size, regex)
      }.toDF("cluster", "messages", "regex")
      .repartition(1)
  }

  private def escapeRegex(s: String) =
    """<([{\^-=$!|]})?*+.>""".foldLeft(s) { (s, c) =>
      s.replace(s"$c", s"\\$c")
    }

  def extractRegex(strings: Seq[String], maxGroups: Option[Int] = None): String = {
    @tailrec
    def recurse(tss: Seq[Seq[String]], maxGroups: Option[Int], acc: Seq[String] = Nil): Seq[String] = {
      if (tss.isEmpty) {
        acc
      } else {
        val (common, rest) = tss.span {_.distinct.size == 1}
        val accWithCommon = acc ++ (common map {_.head} map {
          case "("  => "("
          case ")" => ")"
          case s => escapeRegex(s)
        })
        if (rest.isEmpty) {
          accWithCommon
        } else {
          assert(maxGroups.forall(_ > 0))
          val nextTokens = rest.head
          val matchingPattern = Seq(Digits, Word, AnyString).find { r =>
            val matches = nextTokens.map { s =>
              (r, s, s.matches(r))
            }
            matches.forall(_._3)
          }
          matchingPattern match {
            case None => throw new IllegalArgumentException(s"Unable to find pattern after ${acc.mkString}")
            case Some(p) => recurse(rest.tail, maxGroups map {_ - 1}, accWithCommon :+ p)
          }
        }
      }
    }

    def transpose(ss: Seq[Seq[String]], acc: Seq[Seq[String]] = Nil): Seq[Seq[String]] = {
      if (ss.head.isEmpty) acc.reverse
      else transpose(ss.map {_.tail}, ss.map {_.head} +: acc)
    }

    val tokenizedStrings = strings map tokenize
    val groupedBySize = tokenizedStrings.groupBy(_.size)
    val regexes = groupedBySize map { case (_, tss) => tss.size -> {
      val pattern = recurse(transpose(tokenizedStrings), maxGroups).mkString
      val regex = s"^$pattern$$"
      regex
    }
    }

    val distinctRegexes = regexes.groupBy {_._2}.mapValues(_.keys.sum)
    distinctRegexes.maxBy(_._2)._1
  }

  private val nonDelimitingRegexChars = """\^-=$!|?*+."""
  private def escapeNonDelimitingRegexChars(s: String) =
    nonDelimitingRegexChars.foldLeft(s) { case (s, c) =>
      s.replace(s"$c", s"""\$c""")
    }

  private def agglomerate(vectors: Seq[SampleMessage], maxGroups: Int): (Seq[SampleMessage], (String, Seq[SampleMessage])) = {
    val SampleMessage(f1, _, s) = vectors.head
    val maxDist = 1 << maxGroups
    val closest = vectors.tail.find { case SampleMessage(f2, _, s2) =>
      Vectors.sqdist(f1, f2) <= math.min(maxDist, math.pow(2, f1.numActives - maxDist)) && Try {extractRegex(Seq(s, s2), Some(maxGroups))}.isSuccess
    }
    closest match {
      case None =>
        val rest: Seq[SampleMessage] = vectors.tail
        val re: String = s
        val found: Seq[SampleMessage] = vectors.take(1)
        (rest, (re, found))
      case Some(SampleMessage(_, _, s2)) =>
        val r = extractRegex(Seq(s, s2), Some(maxGroups))
        val (close, far) = vectors.par.partition {
          case SampleMessage(f2, _, s3) => Vectors.sqdist(f1, f2) <= math.min(maxDist, math.pow(2, f1.numActives - 2)) && s3.matches(r)
        }
        val rest: List[SampleMessage] = far.toList
        val re: String = r
        val found: List[SampleMessage] = close.toList
        (rest, (re, found))
    }
  }

  @tailrec
  final def collect(vectors: Seq[SampleMessage], maxGroups: Int, acc: Seq[(String, Seq[SampleMessage])] = Nil): Seq[(String, Seq[SampleMessage])] = {
    if (vectors.isEmpty) acc
    else {
      val (rest, (re, found)) = agglomerate(vectors, maxGroups)

      if (found.size > 10)
        println(f"""${re.take(90)}%-100s found ${found.size}%,7d\tremaining ${rest.size}%7d""")
      collect(rest.toList, maxGroups, (re, found) +: acc)
    }
  }

  final def agglomerateClusters(vectors: Seq[SparseVector], maxGroups: Int = 1): Seq[VectorCluster] = {

    @tailrec
    def recurse(clusters: ParIterable[VectorCluster], groups: Int, acc: Seq[VectorCluster] = Nil, foundNew: Boolean = false): (Seq[VectorCluster], Boolean) = {
      if (clusters.isEmpty) {
        (acc, foundNew)
      } else {
        val first = clusters.head
        val words = first.words
        val maxDist = math.min(1 << groups, words - 1)
        val (t1, t2) = clusters.tail.span(c => c.maxDist(first) > maxDist || c.words != first.words)
        val (rest, found, foundPairs) = if (t2.nonEmpty) {
          (t1 ++ t2.tail, VectorCluster(first, t2.head), true)
        } else {
          (t1, first, foundNew)
        }
        recurse(rest, groups, found +: acc, foundPairs)
      }
    }

    @tailrec
    def agglomerate(clusters: ParIterable[VectorCluster], groups: Int): ParIterable[VectorCluster] = {
      val byWords = clusters.groupBy(_.words) map { case (_, clusters) =>
        recurse(clusters, groups)
      }

      val (newClusters, foundPairs) = (byWords.flatMap {_._1}, byWords.exists(_._2))

      if (foundPairs) agglomerate(newClusters, groups)
      else newClusters
    }

    val clusters = vectors.par.map { v => VectorCluster(v.numActives, v, Nil) }

    @tailrec
    def agglomerateClosest(lusters: ParIterable[VectorCluster], groups: Int): ParIterable[VectorCluster] = {
      val currentLevel = agglomerate(clusters, groups)
      if (groups >= maxGroups) currentLevel
      else agglomerateClosest(currentLevel, groups + 1)
    }

    agglomerateClosest(clusters, 1).toList
  }
}

case class VectorCluster(words: Int, centroid: SparseVector, elements: Seq[VectorCluster]) {
  def centerDist(that: VectorCluster): Double = Vectors.sqdist(centroid, that.centroid)
  def maxDist(that: VectorCluster): Double = distances(that).max
  def minDist(that: VectorCluster): Double = distances(that).min
  private def distances(that: VectorCluster): Iterator[Double] = {
    for {
      v1 <- vectors.iterator
      v2 <- that.vectors.iterator
    } yield (Vectors.sqdist(v1, v2))
  }
  def radius: Double = if (elements.isEmpty) 0 else elements.map {centerDist}.max
  def diameter: Double = if (elements.isEmpty) 0 else maxDist(this)
  def +(that: VectorCluster) = VectorCluster(Seq(this, that))
  def size: Int = if (elements.isEmpty) 1 else elements.map {_.size}.sum
  def vectors: Seq[SparseVector] = if (elements.isEmpty) Seq(centroid) else elements.flatMap {_.vectors}
}

object VectorCluster {
  def apply(c1: VectorCluster, c2: VectorCluster): VectorCluster =
    VectorCluster(Seq(c1, c2))

  def apply(clusters: Seq[VectorCluster]): VectorCluster = {
    require(clusters.map {_.words}.distinct.size == 1)
    VectorCluster(clusters.head.words, mean(clusters.map {_.centroid}), clusters)
  }

  private def mean(vectors: Seq[SparseVector]): SparseVector = {
    val m = scala.collection.mutable.Map.empty[Int, Double]

    vectors foreach { v =>
      v.foreachActive { case (i, v) =>
        m.update(i, m.getOrElse(i, 0d) + v)
      }
    }

    val size = vectors.head.size
    val numberOfVectors = vectors.size
    val (keys, values) = m.toArray.sortBy {_._1}.unzip
    new SparseVector(size, keys, values map {_ / numberOfVectors})
  }

  private def mean(vectors: Seq[DenseVector]): Vector = {
    val s = scala.collection.mutable.Map.empty[Int, Double]

    vectors foreach { v =>
      v.foreachActive { case (i, v) =>
        s.update(i, s.getOrElse(i, 0d) + v)
      }
    }
    val numberOfVectors = vectors.size
    val m = s.mapValues(_ / numberOfVectors).filter(_._2 > 1e-10)

    val size = vectors.head.size
    val (keys, values) = m.toArray.sortBy {_._1}.unzip
    new SparseVector(size, keys, values)
  }
}

case class SampleMessage(features_18: SparseVector, raw_item_id: Long, message: String)
