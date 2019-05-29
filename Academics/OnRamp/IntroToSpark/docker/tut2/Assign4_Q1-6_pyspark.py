
import csv
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("Assignment4").getOrCreate()
sc = spark.sparkContext

# load data to dataframe
path = 'fake_data.csv'
df = spark.read.format('csv').option('header','true').load(path)

# cast income as an integer
df = df.withColumn('Income', df['Income'].cast(IntegerType())) 


# Question 1
print('*' * 30)
print('\nQuestion 1\n')


sqlDf.show()
print('\n\n')


# Question 2
print('*' * 30)
print('\nQuestion 2\n')

sqlDf.show()
print('\n\n')


# Question 3
print('*' * 30)
print('\nQuestion 3\n')

sqlDf.show()
print('\n\n')


# Question 4
print('*' * 30)
print('\nQuestion 4\n')

sqlDf.show()
print('\n\n')


# Question 5
print('*' * 30)
print('\nQuestion 5\n')

sqlDf.show()
print('\n\n')


# Question 6
print('*' * 30)
print('\nQuestion 6\n')

sqlDf.show()
print('\n\n')

