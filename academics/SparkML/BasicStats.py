# imports and tools
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import SparkSession
import os
import numpy as np

# create SparkContext object
spark = SparkSession.builder.appName("Unit03_BasicStats").getOrCreate()
sc = spark.sparkContext


# summary statistics
mat = sc.parallelize(
    [np.array([1.0, 10.0, 100.0]), np.array([2.0, 20.0, 200.0]), np.array([3.0, 30.0, 300.0])]
) 

summary = Statistics.colStats(mat)
print(summary.mean())
print(summary.variance())
print(summary.numNonzeros())

## correlation
# vectors
seriesX = sc.parallelize([1.0, 2.0, 3.0, 3.0, 5.0])
seriesY = sc.parallelize([11.0, 22.0, 33.0, 33.0, 55.0])

print('Pearson correlation is: {}'.format(Statistics.corr(seriesX, seriesY, method = 'pearson')))
print('Spearman correlation is: {}'.format(Statistics.corr(seriesX, seriesY, method = 'spearman')))

# matrix
print('Correlation of matrix: {}'.format(Statistics.corr(mat, method = 'pearson')))


## sampling
# sampling methods can be performed on RDD's of key-value pairs
data = sc.parallelize([
    (1, 'a')
    ,(1, 'b')
    ,(2, 'c')
    ,(2, 'd')
    ,(2, 'e')
    ,(3, 'f')
])

fractions = {1 : 0.1, 2 : 0.6, 3 : 0.3}
approxSample = data.sampleByKey(False, fractions)

## hypothesis testing
from pyspark.mllib.linalg import Matrices, Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics

vec = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25)
# compute goodness of fit. either compare two vectors to each other or compare one vector to a uniform distribution
goodnessOfFitTestResults = Statistics.chiSqTest(vec)
print(goodnessOfFitTestResults)

# pearson's independence test on a matrix
mat = Matrices.dense(3, 2, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
independenceTestResults = Statistics.chiSqTest(mat)
print(independenceTestResults)

# a contingency table can be constructed from an RDD of LabeledPoint/vector pairs. The resulting test returns
# a Chi-squared test results for every feature against the label
obs = sc.parallelize([
    LabeledPoint(1.0, [1.0, 0.0, 3.0])
    ,LabeledPoint(1.0, [1.0, 2.0, 0.0])
    ,LabeledPoint(1.0, [-1.0, 0.0, -0.5])
])
featureTestResults = Statistics.chiSqTest(obs)

for i, result in enumerate(featureTestResults):
    print('column {0}: \n {1}'.format(i, result))


## random data generation
from pyspark.mllib.random import RandomRDDs

# generate a random RDD that contains a million iid values drawn from a normal distribution N(0, 1)
# distribute evenly to 10 partitions
u = RandomRDDs.normalRDD(sc, size = 1000000, numPartitions = 10)
print(u.take(20))

# apply a transformation to return a random RDD that follow a normal distribution N(1, 4)
v = u.map(lambda x: 1.0 + 2.0 * x)
print(v.take(20))

