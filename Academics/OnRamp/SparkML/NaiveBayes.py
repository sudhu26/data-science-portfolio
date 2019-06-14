# imports and tools
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import os
import numpy as np

# create SparkContext object
spark = SparkSession.builder.appName("Unit04_NaiveBayes").getOrCreate()
sc = spark.sparkContext

# PySpark's implementation of Naive Bayes uses a multinomial approach (if needed). it takes in a RDD
# through a LabeledPoint object. It optionally can take a smoothing parameter lambda

data = sc.parallelize([
    LabeledPoint(0.0, [0.0, 0.0])
    ,LabeledPoint(1.0, [0.3, 0.3])
    ,LabeledPoint(1.0, [0.7, 0.3])
    ,LabeledPoint(0.0, [0.1, 0.05])
])

# train model
model = NaiveBayes.train(data, lambda_ = 1.0)

# predict a single sample
prediction = model.predict([0.6, 0.4])

print('Prediction: {}'.format(prediction))
