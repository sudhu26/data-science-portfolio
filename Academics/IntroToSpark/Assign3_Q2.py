
from pyspark.sql import HiveContext, SparkSession
import json

spark = SparkSession.builder.appName("Assignment3_Q2").getOrCreate()
sc = spark.sparkContext
hiveCtx = HiveContext(sc)

path = '/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/examples/src/main/resources/people.json'

peopleDf2 = hiveCtx.read.json(path)

peopleDf2.printSchema()
peopleDf2.registerTempTable('peopleDf2')

names = hiveCtx.sql('SELECT DISTINCT name FROM peopleDf2')

print(names.head(5))
