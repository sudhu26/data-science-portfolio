
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
print(df.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda vals: len(set(vals))).sortBy(lambda a: a[1], ascending = False).take(1))
print('\n\n')


# Question 2
print('*' * 30)
print('\nQuestion 2\n')
print(df.rdd.filter(lambda v: v[1] == 'United States of America').map(lambda x: (x[1], x[4])).groupByKey().mapValues(lambda x: sum(x) / len(x)).collect())
print('\n\n')


# Question 3
print('*' * 30)
print('\nQuestion 3\n')
print(df.rdd.filter(lambda v: v[4] > 100000).filter(lambda v: v[7] == 'FALSE').count())
print('\n\n')


# Question 4
print('*' * 30)
print('\nQuestion 4\n')
print(df.rdd.filter(lambda v: v[1] == 'United States of America').sortBy(lambda x: x[4], ascending = False).map(lambda x: (x[3], x[6], x[4], x[5])).take(10))
print('\n\n')


# Question 5
print('*' * 30)
print('\nQuestion 5\n')
print(df.rdd.groupBy(lambda x: x[5]).count())
print('\n\n')


# Question 6
print('*' * 30)
print('\nQuestion 6\n')
print(df.rdd.filter(lambda v: v[5] == 'Writer').filter(lambda x: x[4] < 100000).count())
print('\n\n')

