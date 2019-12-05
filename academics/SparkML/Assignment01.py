# imports and tools
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
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
print(data.printSchema())

# count the number of data points
print('\nThere are {} observations in the Millionsong dataset'.format(data.count()))

# print the first 40 instances
print('\nHere are the first 40 instances:\n\n')
print(data.show(40))

####################################################################################
## part 2
print('*' * 100)
print('Part 2 - Normalize features between 0 and 1\n')

# assemble features values into a vector and create a feature containing those vectors
assembler = VectorAssembler().setInputCols(data.columns[1:]).setOutputCol('features')
transformed = assembler.transform(data)

# create scaler object, transform feature vectors and add scaledFeatures column 
scaler = MinMaxScaler(inputCol = 'features', outputCol = 'scaledFeatures')
scalerModel = scaler.fit(transformed.select('features'))
scaledData = scalerModel.transform(transformed)

print('Features scaled to range: {} to {}'.format(scaler.getMin(), scaler.getMax()))
# print(scaledData.select('_c0','features','scaledFeatures').show(10))

# limit dataset to label and scaled vectors
scaledData = scaledData.select('_c0','scaledFeatures')

# rename columns
scaledData = scaledData.withColumnRenamed('_c0', 'label').withColumnRenamed('scaledFeatures', 'features')
print(scaledData.show(5))

####################################################################################
## part 3
print('*' * 100)
print('Part 3 - \n')

labelMin = scaledData.select(F.min('label')).collect()[0][0]
labelMax = scaledData.select(F.max('label')).collect()[0][0]

print('Year min: {}'.format(labelMin))
print('Year max: {}'.format(labelMax))

scaledData = scaledData.withColumn('label', F.col('label') - labelMin)
print(scaledData.show(5))

####################################################################################
## part 4
print('*' * 100)
print('Part 4 - \n')

# split dataset into train, validation and test sets
(train_data, validationData, test_data) = scaledData.randomSplit([0.7, 0.2, 0.1], seed = 0)
train_data.cache()
validationData.cache()
test_data.cache()

train_dataCount = train_data.count()
validationDataCount = validationData.count()
test_dataCount = test_data.count()

print('Full dataset size: {}'.format(scaledData.count()))
print('Training dataset size: {}'.format(train_dataCount))
print('Validation dataset size: {}'.format(validationDataCount))
print('Test dataset size: {}'.format(test_dataCount))

print('Training + Validation + Test  = {}'.format(train_dataCount + validationDataCount + test_dataCount))

# baseline model that returns average shifted year irrespective of the output
baselineTrainYear = train_data.select(F.mean('label')).collect()[0][0]
print('Baseline model prediction: {}'.format(baselineTrainYear))

# create functions that calculate root mean squared error
def squareError(label, prediction):
    return (label - prediction) * (label - prediction)

def rootMeanSquaredError(labelPredPairs):
    return np.sqrt(labelPredPairs.map(lambda x: squareError(x[0], x[1])).mean())

perfectInput = sc.parallelize([(1, 1), (2, 2), (3, 3)]) 
imperfectInput = sc.parallelize([(2, 1), (4, 2), (4, 3)]) 

print('Perfect predictions: {}'.format(rootMeanSquaredError(perfectInput)))
print('Imperfect predictions: {}'.format(rootMeanSquaredError(imperfectInput)))

# measure performance using baseline model
test_dataRDD = test_data.rdd
labelPredPairsTest = test_dataRDD.map(lambda x: (x.label, baselineTrainYear))
print('Baseline performance (test): {}'.format(rootMeanSquaredError(labelPredPairsTest)))

# ####################################################################################
# ## part 5
# visualize actual vs. prediction
act = labelPredPairsTest.map(lambda x: x[0]).collect() 
pred = labelPredPairsTest.map(lambda x: x[1]).collect() 

plt.scatter(act, pred, alpha = 0.1, s = 2)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.axis([0, 80, 0, 100])
plt.savefig('ActualVsPred.png')
