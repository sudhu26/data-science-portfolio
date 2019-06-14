# imports and tools
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F

import os
import numpy as np

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
data = spark.read.load(path , format = 'csv',header = 'False', inferschema = 'true', sep = ",")
print(data.printSchema())

# count the number of data points
print('\nThere are {} observations in the Millionsong dataset'.format(data.count()))

# print the first 40 instances
print('\nHere are the first 40 instances:\n\n')
# print(data.show(40))
print(data.show(5))

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
print(scaledData.select('_c0','features','scaledFeatures').show(10))

# limit dataset to label and scaled vectors
scaledData = scaledData.select('_c0','scaledFeatures')

# rename columns
scaledData = scaledData.withColumnRenamed('_c0', 'label').withColumnRenamed('scaledFeatures', 'features')
print(scaledData.show(10))

####################################################################################
## part 3
print('*' * 100)
print('Part 3 - \n')

# turn dataframe into RDD
scaledData = scaledData.rdd
print(type(scaledData))

labels = scaledData.map(lambda x: x.label)

labelMin = labels.min()
labelMax = labels.max()

print('Min: {}'.format(labelMin))
print('Max: {}'.format(labelMax))




# print('Min: {}'.format(scaledData.select(F.min('_c0')).collect()))
# print('Max: {}'.format(scaledData.select(F.max('_c0')).collect()))

# minum = (F.col('_c0') - 1922)

# scaledData = scaledData.withColumn('_c0', minum)
# print(scaledData.show(5))

# ####################################################################################
# ## part 4
# print('*' * 100)
# print('Part 4 - \n')

# # split dataset into train, validation and test sets
# (train, test, validation) = scaledData.randomSplit([0.7, 0.2, 0.1], seed = 0)

# # print('Full dataset size: {}'.format(scaledData.count()))
# # print('Training dataset size: {}'.format(train.count()))
# # print('Test dataset size: {}'.format(test.count()))
# # print('Validation dataset size: {}'.format(validation.count()))

# # create model
# lr = LinearRegression(featuresCol = 'scaledFeatures', labelCol = '_c0', maxIter = 10)
# lrModel = lr.fit(train)
# print('Coefficients:\n\t{}'.format(str(lrModel.coefficients)))
# print('Intercept:\n\t{}'.format(str(lrModel.intercept)))

# trainingSummary = lrModel.summary
# print('RMSE:\n\t{}'.format(str(trainingSummary.rootMeanSquaredError)))
# print('R-squared:\n\t{}'.format(str(trainingSummary.r2)))

# # predictions
# lrPredictions = lrModel.transform(test)
# lrPredictions.select('prediction','_c0','scaledFeatures').show(5)

# lrEvaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = '_c0', metricName = 'rmse')
# rmse = lrEvaluator.evaluate(lrPredictions)
# print('RMSE on test data: {}'.format(rmse))

# ####################################################################################
# ## part 5
# print('*' * 100)
# print('Part 5 - \n')

# # https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8122459673715921/2786992496259623/2531719484635850/latest.html
# https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a