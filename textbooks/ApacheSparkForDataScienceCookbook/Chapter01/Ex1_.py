"""
Chapter 1 
Example 1 - Initializing SparkContext

SparkContext is created on the driver and connects with the cluster. Only one SparkContext is
created per application, so streaming applications and Spark SQL applications are created using
StreamingContext and SQLContext on top of SparkContext. 

Initially, RDDs are created using SparkContext
"""

from pyspark import SparkContext

stocks = 'hdfs://namenode:9000/stocks.txt'

sc = SparkContext('', 'ApplicationName')
data - sc.textFile(stocks)

totalLines = data.count()
print('Total lines: {}'.format(totalLines))
