# imports and tools
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import os
import numpy as np

# create SparkContext object
spark = SparkSession.builder.appName("Unit04_DecisionTrees").getOrCreate()
sc = spark.sparkContext

data = MLUtils.loadLibSVMFile(sc, '{}/data/mllib/sample_libsvm_data.txt'.format(os.environ['SPARK_HOME']))
data.take(1)

# set aside test data
(trainingData, test_data) = data.randomSplit([0.7, 0.3])

## standard decicion tree
# build decision tree model
model = DecisionTree.trainClassifier(trainingData
                                    ,numClasses = 2
                                    ,objectFeaturesInfo = {}
                                    ,impurity = 'gini'
                                    ,maxDepth = 5
                                    ,maxBins = 32
    )

# prediction and test error
predictions = model.predict(test_data.map(lambda x: x.features))
labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)

# count incorrect
testError = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
print('test error: {}'.format(testError))
print(model.toDebugString())

# save model
model.save(sc, './decisionTreeModel')
# reload = DecisionTreeModel.load(sc, './decisionTreeModel')

## random forest
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

# build model
model = RandomForest.trainClassifier(trainingData
                                    ,numClasses = 2
                                    ,objectFeaturesInfo = {}
                                    ,numTrees = 3
                                    ,featureSubsetStrategy = 'auto'
                                    ,impurity = 'gini'
                                    ,maxDepth = 4
                                    ,maxBins = 32
    )

# prediction and test error
predictions = model.predict(test_data.map(lambda x: x.features))
labelsAndPredictions = test_data.map(lambda lp: lp.label).zip(predictions)

# count incorrect
testError = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
print('test error: {}'.format(testError))
print(model.toDebugString())

# save model
model.save(sc, './randomForestModel')
# reload = DecisionTreeModel.load(sc, './randomForestModel')

