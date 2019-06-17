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



####################################################################################
## part 2
# 
print('*' * 100)
print('Part 1 - \n')


####################################################################################
## part 2
# 
print('*' * 100)
print('Part 1 - \n')


####################################################################################
## part 3
# 
print('*' * 100)
print('Part 3 - \n')


####################################################################################
## part 4
# 
print('*' * 100)
print('Part 5 - \n')


####################################################################################
## part 2
# 
print('*' * 100)
print('Part 6 - \n')

