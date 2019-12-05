# imports and tools
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

import os
import numpy as np

import matplotlib.pyplot as plt

# create SparkContext object
spark = SparkSession.builder.appName("Assignment_01").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

####################################################################################
## part 1
# load data to dataframe
print('*' * 100)
print('Part 1 - load data into dataframe\n')
path = 'MSD.txt'
data = spark.read.load(path , format = 'csv', header = 'False', inferschema = 'true', sep = ",")

# rename label column
features = data.columns[1:]
data = data.select(F.col('_c0').alias('label'), *features)

# shift label by subtracting minimum
labelMin = data.select(F.min('label')).collect()[0][0]
data = data.withColumn('label', F.col('label') - labelMin)

# review
print(data.printSchema())

# split dataset into train, validation and test sets
(train_data, validationData, test_data) = data.randomSplit([0.7, 0.1, 0.2], seed = 0)
train_data.cache()
validationData.cache()
test_data.cache()

train_dataCount = train_data.count()
validationDataCount = validationData.count()
test_dataCount = test_data.count()

print('Full dataset size: {}'.format(data.count()))
print('Training dataset size: {}'.format(train_dataCount))
print('Validation dataset size: {}'.format(validationDataCount))
print('Test dataset size: {}'.format(test_dataCount))

print('Training + Validation + Test  = {}'.format(train_dataCount + validationDataCount + test_dataCount))

####################################################################################
## part 2
print('*' * 100)
print('Part 2 - Train the model and evaluate on validation dataset \n')

# data processing pipeline
assembler = VectorAssembler(inputCols = features, outputCol = 'unscaledFeatures')
minMaxScaler = MinMaxScaler(inputCol = 'unscaledFeatures', outputCol = 'features')
stages = [assembler, minMaxScaler]
pipeline = Pipeline(stages = stages)

procPipeline = pipeline.fit(train_data)
train_data = procPipeline.transform(train_data)
validationData = procPipeline.transform(validationData)
test_data = procPipeline.transform(test_data)

train_data = train_data.select('label','features')
validationData = validationData.select('label','features')
test_data = test_data.select('label','features')

# train model and evaluate on validation data
lr = LinearRegression(maxIter = 100)
model = lr.fit(train_data)
valPreds = model.transform(validationData)

# evaluate
# create functions that calculate root mean squared error
def squareError(label, prediction):
    return (label - prediction) * (label - prediction)

def rootMeanSquaredError(labelPredPairs):
    return np.sqrt(labelPredPairs.map(lambda x: squareError(x[0], x[1])).mean())

# evaluate performance on validation set
valPredsRDD = valPreds.rdd
valuesAndPredsVal = valPredsRDD.map(lambda x: (x.label, x.prediction))
print('Validation RMSE: {}'.format(rootMeanSquaredError(valuesAndPredsVal)))

####################################################################################
## part 3
print('*' * 100)
print('Part 3 - Visualize the log of the training error\n')

# convert data sets
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

train_dataRDD = train_data.rdd
train_dataRDD = train_dataRDD.map(lambda x: LabeledPoint(x[0], MLLibVectors.fromML(x[1])))
train_dataRDD.persist()

numIters = 50
errors = []
for i in range(1, numIters + 1):
    model = LinearRegressionWithSGD.train(train_dataRDD
                                          ,iterations = i
                                          ,step = 0.01
                                          )
    valuesAndPredsTrain = train_dataRDD.map(lambda x: (x.label, model.predict(x.features)))   
    errors.append(rootMeanSquaredError(valuesAndPredsTrain))
    print(errors)

# visualize actual vs. prediction
x = np.arange(1, numIters + 1)
y = np.log(errors)

plt.scatter(x, y, alpha = 1.0, s = 2.5)
plt.xlabel('iteration')
plt.ylabel('log RMSE')
plt.axis([0, 50, 0, 5])
plt.savefig('ErrorOverTime.png')


####################################################################################
## part 4
print('*' * 100)
print('Part 4 - \n')

# evaluate performance of model on test set
test_dataRDD = test_data.rdd
test_dataRDD = test_dataRDD.map(lambda x: LabeledPoint(x[0], MLLibVectors.fromML(x[1])))  
test_dataRDD.persist()

valuesAndPredsTest = test_dataRDD.map(lambda x: (x.label, model.predict(x.features)))
print('Test RMSE: {}'.format(rootMeanSquaredError(valuesAndPredsTest)))

