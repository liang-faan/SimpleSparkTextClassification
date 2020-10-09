import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, split}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object SMSClassification {
  final val VECTOR_SIZE = 100;

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder()
      .appName("SMS Classification")
      .master("local[*]")
      .getOrCreate();

    val textSchema = StructType(Array(
      StructField("label", StringType, nullable = false),
      StructField("message", StringType, nullable = false)
    ));

    val inputFile = "./smsspamcollection/SMSSpamCollection";
    var properties = Map("header" -> "false", "inferSchema" -> "false", "delimiter" -> "\\t")
    val df = sparkSession.read
      .options(properties)
      .schema(textSchema)
      .csv(inputFile);

//    df.show(10);

    val trainingDf=df.select(col("label"), split(col("message")," ").as("message"));

    trainingDf.show(10)

    val labelIndex = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainingDf);

//    print(labelIndex.labels);

    val word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(VECTOR_SIZE).setMinCount(1)

    val layers = Array[Int](VECTOR_SIZE, 6, 5, 2);

    val mlpc = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(512)
      .setSeed(1234L)
      .setMaxIter(128)
      .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction");

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictionLabel")
      .setLabels(labelIndex.labels);

    val Array(trainingData, testData) = trainingDf.randomSplit(Array(0.8, 0.2));
    val pipeline = new Pipeline()
      .setStages(Array(labelIndex, word2Vec, mlpc, labelConverter));
    val model = pipeline.fit(trainingData);
    model.save("./smsspamcollection.model")

    val predictionResultDf = model.transform(testData);

//    predictionResultDf.printSchema();
//    predictionResultDf.show(100);

    predictionResultDf.filter(col("predictionLabel").equalTo("spam")).show(100)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val predictionAccuracy = evaluator.evaluate(predictionResultDf);
    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
  }
}
