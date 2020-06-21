// Databricks notebook source
val df = spark.read.format("csv")
  .option("sep", ",")
  .option("inferSchema", "true")
  .option("header", "true")
  .load("/titanic/train.csv")

val df_train = df.drop("PassengerId").drop("Name").drop("Ticket").drop("Cabin")

display(df_train)

// COMMAND ----------

val df = spark.read.format("csv")
  .option("sep", ",")
  .option("inferSchema", "true")
  .option("header", "true")
  .load("/titanic/test.csv")

val df_test = df.drop("PassengerId").drop("Name").drop("Ticket").drop("Cabin")

display(df_test)

// COMMAND ----------

// MAGIC %md ### Import AutoML components

// COMMAND ----------

import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference
import com.databricks.labs.automl.params.MLFlowConfig
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %md ### Configure AutoML

// COMMAND ----------

// Generic configuration
val experimentNamePrefix = "/Users/adelsors@hotmail.com/Titanic"
val RUNVERSION = 1
val labelColumn = "Survived"
val xgBoostExperiment = s"runXG_$RUNVERSION"
val logisticRegExperiment = s"runLG_$RUNVERSION"
val projectName = "AutoML_Demo"

// This is the configuration of the hardware available
val nodeCount = 4
val coresPerNode = 4
val totalCores = nodeCount * coresPerNode
val driverCores = 4

val mlFlowTrackingURI = ""

// COMMAND ----------

// MAGIC %md ### Configure Overrides for Xgboost and LogisticRegression
// MAGIC An important aspect of the AutoML toolkit is the ability to modify the generic maps with your own overrides.  In general, you can start with the defaults and change these as you want more control over how this works as you become more familiar with the toolkit.  For example, we have configured the `tunerTrainSplitMethod -> "stratified"` due to our unbalanced dataset.<br>
// MAGIC 
// MAGIC In this case, we will setup Xgboost and LogisticRegression overridden configurations below so that we can use FamilyRunner API.

// COMMAND ----------

val xgBoostOverrides = Map(
  "labelCol" -> labelColumn,
  "scoringMetric" -> "accuracy",
  "oneHotEncodeFlag" -> true,
  "autoStoppingFlag" -> true,
  "tunerAutoStoppingScore" -> 0.91,
  "tunerParallelism" -> 2, //driverCores * 2,
  "tunerKFold" -> 2,
  "tunerTrainPortion" -> 0.7,
  "tunerTrainSplitMethod" -> "stratified",
  "tunerInitialGenerationMode" -> "permutations",
  "tunerInitialGenerationPermutationCount" -> 8,
  "tunerInitialGenerationIndexMixingMode" -> "linear",
  "tunerInitialGenerationArraySeed" -> 42L,
  "tunerFirstGenerationGenePool" -> 16,
  "tunerNumberOfGenerations" -> 3,
  "tunerNumberOfParentsToRetain" -> 2,
  "tunerNumberOfMutationsPerGeneration" -> 4,
  "tunerGeneticMixing" -> 0.8,
  "tunerGenerationalMutationStrategy" -> "fixed",
  "tunerEvolutionStrategy" -> "batch",
  "tunerHyperSpaceInferenceFlag" -> true,
  "tunerHyperSpaceInferenceCount" -> 400000,
  "tunerHyperSpaceModelType" -> "XGBoost",
  "tunerHyperSpaceModelCount" -> 8,
  "mlFlowLoggingFlag" -> false,
  "mlFlowLogArtifactsFlag" -> false,
  "mlFlowTrackingURI" -> "https://demo.cloud.databricks.com",
  "mlFlowExperimentName" -> s"$experimentNamePrefix/$projectName/$xgBoostExperiment",
  //"mlFlowAPIToken" -> dbutils.notebook.getContext().apiToken.get,
  "mlFlowLoggingMode" -> "bestOnly",
  "mlFlowBestSuffix" -> "_best",
  "pipelineDebugFlag" -> true
)

val logisticRegOverrides = Map(
  "labelCol" -> labelColumn,
  "scoringMetric" -> "accuracy",
  "oneHotEncodeFlag" -> true,
  "autoStoppingFlag" -> true,
  "tunerAutoStoppingScore" -> 0.91,
  "tunerParallelism" -> 2, //driverCores * 2,
  "tunerKFold" -> 2,
  "tunerTrainPortion" -> 0.7,
  "tunerTrainSplitMethod" -> "stratified",
  "tunerInitialGenerationMode" -> "permutations",
  "tunerInitialGenerationPermutationCount" -> 8,
  "tunerInitialGenerationIndexMixingMode" -> "linear",
  "tunerInitialGenerationArraySeed" -> 42L,
  "tunerFirstGenerationGenePool" -> 16,
  "tunerNumberOfGenerations" -> 3,
  "tunerNumberOfParentsToRetain" -> 2,
  "tunerNumberOfMutationsPerGeneration" -> 4,
  "tunerGeneticMixing" -> 0.8,
  "tunerGenerationalMutationStrategy" -> "fixed",
  "tunerEvolutionStrategy" -> "batch",
  "mlFlowLoggingFlag" -> false,
  "mlFlowLogArtifactsFlag" -> false,
  "mlFlowTrackingURI" -> mlFlowTrackingURI,
  "mlFlowExperimentName" -> s"$experimentNamePrefix/$projectName/$logisticRegExperiment",
  //"mlFlowAPIToken" -> dbutils.notebook.getContext().apiToken.get,
  "mlFlowLoggingMode" -> "bestOnly",
  "mlFlowBestSuffix" -> "_best",
  "pipelineDebugFlag" -> true
)

// COMMAND ----------

// MAGIC %md ### Family Runner API
// MAGIC Using FamilyRunner API, determine the best possible SparkML PipelineModel for an ML task by supplying a DataFrame and an Array of InstanceConfig objects that have been defined with ConfigurationGenerator

// COMMAND ----------

val xgBoostConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "classifier", xgBoostOverrides)
val logisticRegressionConfig = ConfigurationGenerator.generateConfigFromMap("LogisticRegression", "classifier", logisticRegOverrides)

val runner = FamilyRunner(df_train, Array(xgBoostConfig, logisticRegressionConfig)).executeWithPipeline()

// COMMAND ----------

// MAGIC %md ## Pipeline API for inference
// MAGIC Based on the scoring optimization strategy set, pipeline API returns the best pipeline model as well as a corresponding best MLflowRun ID. Check [AutoML pipeline API docs](https://github.com/databrickslabs/automl-toolkit/blob/master/PIPELINE_API_DOCS.md) for additional details.
// MAGIC We can run inference using Mlflow Run ID or using PipelineModel

// COMMAND ----------

// MAGIC %md ### Inference using PipelineModel with LogisticRegression
// MAGIC When MLflow is disabled, it is possible to run inference directly using returned PipelineModel.<br>
// MAGIC Note: Pipeline API internally adds a ```IndexToString``` stage so that the Prediction column contains original labels 

// COMMAND ----------

val pipelineModel = runner.bestPipelineModel("LogisticRegression")
val df = df_test.withColumn("Survived", expr("0"))
val dfres = pipelineModel.transform(df).drop("features")
//display(dfres)
//dfres.select("prediction").coalesce(1).write.mode("overwrite").csv("/titanic/results-automl-toolkit-lr.csv")

// COMMAND ----------

var initSeq: Int=892;
val dflr = dfres.select("prediction")
display(dflr)
dflr.coalesce(1).write.mode("overwrite").csv("/titanic/results-automl-toolkit-xg.csv")

// COMMAND ----------

val pipelineModel = runner.bestPipelineModel("XGBoost")
val df = df_test.withColumn("Survived", expr("0"))
val dfres = pipelineModel.transform(df).drop("features")
//display(pipelineModel.transform(df).drop("features"))
//dfres.select("prediction").coalesce(1).write.mode("overwrite").csv("/titanic/results-automl-toolkit-xg.csv")

// COMMAND ----------

val dfxg = dfres.select("prediction")
display(dfxg)
dfxg.coalesce(1).write.mode("overwrite").csv("/titanic/results-automl-toolkit-xg.csv")

// COMMAND ----------

// MAGIC %md ### Feature Engineering with Pipeline API
// MAGIC It is also possible to use Pipeline API for running only feature engineering tasks based on an array of selected configurations

// COMMAND ----------

val featureEngPipelineModel = FamilyRunner(df_train, Array(xgBoostConfig, logisticRegressionConfig)).generateFeatureEngineeredPipeline(verbose=true)("XGBoost")
val featuredData = featureEngPipelineModel.transform(df_train)
display(featuredData)

// COMMAND ----------

// MAGIC %md ## MLFlow Integration
// MAGIC In addition to existing parameters and metrics logged, AutoML Toolkit's pipeline API internally logs a lot of useful information to MLflow using Tags API. At any given point of pipeline execution, we can look under MLflow Run Id to find the latest status of training progress under ```PipelineExecutionCurrentStatus``` tag. Once a run is complete, a full log of pipeline execution is logged, including the total number of pipeline stages executed.
// MAGIC <br/>
// MAGIC <img src="https://raw.githubusercontent.com/databrickslabs/automl-toolkit/master/images/mlflow-1.png" width="1600"/>
// MAGIC <br/>
// MAGIC &nbsp;

// COMMAND ----------

// MAGIC %md ### Pipeline Debug Mode On
// MAGIC As mentioned in the [pipeline API documentation](https://github.com/databricks/providentia/blob/master/PIPELINE_API_DOCS.md#pipeline-configurations), when pipeline debug flag is turned on, AutoML toolkit will log each pipeline stage transformations metadata under Tags prefixed with ```pipeline_stage_{Pipeline_STAGE_CLASS_NAME}```. As can be seen in an example below, [CardinalityLimitColumnPrunerTransformer](https://github.com/databrickslabs/automl-toolkit/blob/master/src/main/scala/com/databricks/labs/automl/pipeline/CardinalityLimitColumnPrunerTransformer.scala) debugs all params and spark trasnformations in and out of that pipeline stage
// MAGIC <br/>
// MAGIC <img src="https://raw.githubusercontent.com/databrickslabs/automl-toolkit/master/images/mlflow-2.png" width="1600"/>
// MAGIC <br/>
