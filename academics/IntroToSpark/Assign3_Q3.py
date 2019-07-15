
import csv
from pyspark.sql import SparkSession

path = '/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/examples/src/main/resources/people.txt'

with open(path, newline = '') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    for row in data:
        print(', '.join(row))


spark = SparkSession.builder.appName("Assignment3_Q3").getOrCreate()
sc = spark.sparkContext

data = sc.textFile(path)

print(data.first())

print(data.count())
