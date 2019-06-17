
import csv
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("Assignment4").getOrCreate()
sc = spark.sparkContext

# load data to dataframe
path = './fake_data.csv'
df = spark.read.format('csv').option('header','true').load(path)

# cast income as an integer
df = df.withColumn('Income', df['Income'].cast(IntegerType())) 

# create SQL temp table
df.createOrReplaceTempView('df')

# Question 1
print('*' * 30)
print('\nQuestion 1\n')

sqlDf = spark.sql("""
SELECT
     Birth_Country
     ,COUNT(ID)
FROM df
GROUP BY Birth_Country
ORDER BY Count(ID) desc
LIMIT 1
""")

sqlDf.show()
print('\n\n')


# Question 2
print('*' * 30)
print('\nQuestion 2\n')
sqlDf = spark.sql("""
SELECT
     Birth_Country
     ,MEAN(Income)
FROM df
WHERE Birth_Country = 'United States of America'
GROUP BY Birth_Country
""")
sqlDf.show()
print('\n\n')


# Question 3
print('*' * 30)
print('\nQuestion 3\n')
sqlDf = spark.sql("""
SELECT 
    COUNT(ID) 
FROM df 
WHERE Income > 100000 
AND Loan_Approved = 'FALSE'
""")
sqlDf.show()
print('\n\n')


# Question 4
print('*' * 30)
print('\nQuestion 4\n')
sqlDf = spark.sql("""
SELECT 
    * 
FROM df 
WHERE Birth_Country = 'United States of America' 
ORDER BY Income desc 
LIMIT 10 
""")
sqlDf.show()
print('\n\n')


# Question 5
print('*' * 30)
print('\nQuestion 5\n')
sqlDf = spark.sql("""
WITH tmp as (SELECT DISTINCT 
    JOB 
FROM df 
) 
  
SELECT COUNT(JOB) FROM tmp 
""")
sqlDf.show()
print('\n\n')


# Question 6
print('*' * 30)
print('\nQuestion 6\n')
sqlDf = spark.sql("""
WITH tmp as (SELECT 
    ID 
FROM df 
WHERE Income < 100000 
AND Job = 'Writer' 
) 
  
SELECT COUNT(ID) FROM tmp 
""")
sqlDf.show()
print('\n\n')

