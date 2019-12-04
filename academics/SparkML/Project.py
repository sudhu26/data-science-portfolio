# imports and tools
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,  CrossValidator

from sklearn.metrics import confusion_matrix

import os
import numpy as np
import pandas as pd

# create SparkContext object
spark = SparkSession.builder.appName('Project').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext


####################################################################################
## Load data
print()
print('*' * 100)
print('Load data')
print('*' * 100)
print()

path = 's3://tdp-ml-datasets/misc/CreditCardDefault.csv'
data = spark.read.load(path , format = 'csv', header = 'True', inferschema = 'true', sep = ",")

# rename label column and drop ID columns
features = data.columns[1:-1]
data = data.select(F.col('DEFAULT').alias('label'), *features)

# split dataset into train and validation sets
(trainData, validationData) = data.randomSplit([0.8, 0.2], seed = 0)
trainData.cache()
validationData.cache()

# double check that the split performed as expected
print('Full dataset size: {}'.format(data.count()))
print('Training dataset size: {}'.format(trainData.count()))
print('Validation dataset size: {}'.format(validationData.count()))
print('Training + Validation = {}'.format(trainData.count() + validationData.count()))


####################################################################################
## Data transformatoin
print()
print('*' * 100)
print('Data transformation')
print('*' * 100)
print()

## data processing steps
# create label / feature vector representation
assembler = VectorAssembler(inputCols = features, outputCol = 'unscaledFeatures')

# perform min/max scaling on data
minMaxScaler = MinMaxScaler(inputCol = 'unscaledFeatures', outputCol = 'features')

print('\n\n<<COMPLETE>>\n\n')

####################################################################################
## Baseline model
print()
print('*' * 100)
print('Baseline model')
print('*' * 100)
print('*' * 60)
print('Baseline model - learning')
print('*' * 60)
print()

### Baseline model
## basic logistic regression model
lr = LogisticRegression(maxIter = 100, regParam = 1.0)

# create pipeline that includes preprocessing steps and model
stages = [assembler, minMaxScaler, lr]
pipeline = Pipeline(stages = stages)

# fit model using training data
modelBasicLr = pipeline.fit(trainData)

# generate predictions on both training data and validation data
trainPredsBasicLr = modelBasicLr.transform(trainData)
validationPredsBasicLr = modelBasicLr.transform(validationData)

print('\n\n<<COMPLETE>>\n\n')

print()
print('*' * 60)
print('Baseline model - evaluation')
print('*' * 60)
print()

## logistic regression summary
# Understand the nature of the prediction accuracy
print()
print('*' * 30)
print('Baseline model TPR, TNR, FPR, FNR - training data')
print('*' * 30)
print()

fp = trainPredsBasicLr.filter((trainPredsBasicLr['prediction'] == 1) & (trainPredsBasicLr['label'] == 0)).count()
fn = trainPredsBasicLr.filter((trainPredsBasicLr['prediction'] == 0) & (trainPredsBasicLr['label'] == 1)).count()
tp = trainPredsBasicLr.filter((trainPredsBasicLr['prediction'] == 1) & (trainPredsBasicLr['label'] == 1)).count()
tn = trainPredsBasicLr.filter((trainPredsBasicLr['prediction'] == 0) & (trainPredsBasicLr['label'] == 0)).count()
print('True positives: \t{:}'.format(tp))
print('True negatives: \t{:}'.format(tn))
print('False positives: \t{:}'.format(fp))
print('False negatives: \t{:}'.format(fn))
print('Total predictions: \t{:}'.format(trainPredsBasicLr.count()))

# confusion matrix
print('\nConfusion matrix:\n')
yTrueBasicLr = trainPredsBasicLr.select('label')
yTrueBasicLr = yTrueBasicLr.toPandas()
yPredBasicLr = trainPredsBasicLr.select('prediction')
yPredBasicLr = yPredBasicLr.toPandas()

confMatrix = confusion_matrix(yTrueBasicLr, yPredBasicLr, labels = [0, 1])
print(confMatrix)

# metrics by label category
print('*' * 30)
print('Baseline model summary - training data, by label')
print('*' * 30)
print()

model = modelBasicLr.stages[2]
modelSummary = model.summary

print('FPR by label:')
for i, rate in enumerate(modelSummary.falsePositiveRateByLabel):
    print('label {:}:\t {:.9f}'.format(i, rate))
    print()

print('TPR by label:')
for i, rate in enumerate(modelSummary.truePositiveRateByLabel):
    print('label {:}:\t {:.9f}'.format(i, rate))
    print()
    
print('Precision by label:')
for i, prec in enumerate(modelSummary.precisionByLabel):
    print('label {:}:\t {:.9f}'.format(i, prec))
    print()
    
print('Recall by label:')
for i, rec in enumerate(modelSummary.recallByLabel):
    print('label {:}:\t {:.9f}'.format(i, rec))
    print()
    
print('F-measure by label:')
for i, f in enumerate(modelSummary.fMeasureByLabel()):
    print('label {:}:\t {:.9f}'.format(i, f))
    print()

# individual metrics
print()
print('*' * 30)
print('Baseline model - assorted evaluation metrics')
print('*' * 30)
print()
accuracy = modelSummary.accuracy
falsePositiveRate = modelSummary.weightedFalsePositiveRate
truePositiveRate = modelSummary.weightedTruePositiveRate
fMeasure = modelSummary.weightedFMeasure()
precision = modelSummary.weightedPrecision
recall = modelSummary.weightedRecall
auc = modelSummary.areaUnderROC
print("Accuracy:\t{:.9f}\nFPR:\t{:.9f}\nTPR:\t{:.9f}\nF-measure:\t{:.9f}\nPrecision:\t{:.9f}\nRecall:\t{:.9f}\nAUC:\t{:.9f}".format(accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall, auc))
print()

# calculate train and validation accuracy
print()
print('*' * 30)
print('Final baseline model assessment using accuracy')
print('*' * 30)
print()
evaluator = MulticlassClassificationEvaluator(labelCol = 'label'
                                              ,predictionCol = 'prediction'
                                              ,metricName = "accuracy"
                                              )

accuracy_train = evaluator.evaluate(trainPredsBasicLr)
print('Training error: {:.9f}'.format(1.0 - accuracy_train))
accuracyValid = evaluator.evaluate(validationPredsBasicLr)
print('Validation error: {:.9f}'.format(1.0 - accuracyValid))
print()


###################################################################################
## Cross-validation workflow with grid search to find better model than baseline
print()
print('*' * 100)
print('Cross-validated model w/ grid search')
print('*' * 100)
print('*' * 60)
print('Cross-validated model - learning')
print('*' * 60)
print()

##### Cross validation - random forest
# instantiate random forest classifier
rf = RandomForestClassifier(featureSubsetStrategy = 'auto', impurity = 'gini')

# parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 15, 20, 25, 30, 40, 50]) \
    .addGrid(rf.maxDepth, [2, 3, 4, 5, 6, 7, 8]) \
    .addGrid(rf.maxBins, [16, 32, 48]) \
    .build()

# create pipeline that includes preprocessing steps and model
stages = [assembler, minMaxScaler, rf]
pipeline = Pipeline(stages = stages)

# cross validator
cv = CrossValidator(estimator = pipeline
                    ,estimatorParamMaps = paramGrid
                    ,evaluator = BinaryClassificationEvaluator()
                    ,numFolds = 8
    )

# fit model using training data
modelCvRf = cv.fit(trainData)

# display results
params = [{p.name: v for p, v in m.items()} for m in modelCvRf.getEstimatorParamMaps()]
resultsCvRf = pd.DataFrame.from_dict([
    {modelCvRf.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, modelCvRf.avgMetrics)
])
display(resultsCvRf)

# generate predictions on both training data and validation data using best model
bestModel = modelCvRf.bestModel
trainPredsCvRf = bestModel.transform(trainData)
validationPredsCvRf = bestModel.transform(validationData)

print('\n\n<<COMPLETE>>\n\n')

print()
print('*' * 60)
print('Cross-validated model - evaluation')
print('*' * 60)
print()

## logistic regression summary
# Understand the nature of the prediction accuracy
print()
print('*' * 30)
print('Cross-validated model - TPR, TNR, FPR, FNR - training data')
print('*' * 30)
print()

fp = trainPredsCvRf.filter((trainPredsCvRf['prediction'] == 1) & (trainPredsCvRf['label'] == 0)).count()
fn = trainPredsCvRf.filter((trainPredsCvRf['prediction'] == 0) & (trainPredsCvRf['label'] == 1)).count()
tp = trainPredsCvRf.filter((trainPredsCvRf['prediction'] == 1) & (trainPredsCvRf['label'] == 1)).count()
tn = trainPredsCvRf.filter((trainPredsCvRf['prediction'] == 0) & (trainPredsCvRf['label'] == 0)).count()
print('True positives: \t{:}'.format(tp))
print('True negatives: \t{:}'.format(tn))
print('False positives: \t{:}'.format(fp))
print('False negatives: \t{:}'.format(fn))
print('Total predictions: \t{:}'.format(trainPredsCvRf.count()))

# confusion matrix
print('\nCross-validated model - Confusion matrix:\n')
yTrueCvRf = trainPredsCvRf.select('label')
yTrueCvRf = yTrueCvRf.toPandas()
yPredCvRf = trainPredsCvRf.select('prediction')
yPredCvRf = yPredCvRf.toPandas()

confMatrix = confusion_matrix(yTrueCvRf, yPredCvRf, labels = [0, 1])
print(confMatrix)

# calculate train and validation accuracy
print()
print('*' * 30)
print('Final cross-validated model assessment using accuracy')
print('*' * 30)
print()
evaluator = MulticlassClassificationEvaluator(labelCol = 'label'
                                              ,predictionCol = 'prediction'
                                              ,metricName = "accuracy"
                                              )

accuracy_train = evaluator.evaluate(trainPredsCvRf)
print('Training error: {:.9f}'.format(1.0 - accuracy_train))
accuracyValid = evaluator.evaluate(validationPredsCvRf)
print('Validation error: {:.9f}'.format(1.0 - accuracyValid))
print()