
from pyspark.sql import SparkSession
from operator import add

# create spark session and context objects
spark = SparkSession.builder.appName("Assignment2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('WARN')

# create RDD
text_file = sc.textFile('assignment_2_datafile.txt')

# lower case all words and count all words. store as tuple of word/count pairs
counts = text_file.flatMap(lambda x: [(w.lower(), 1) for w in x.split()]).reduceByKey(add)

# sort counts variable descending
counts = sorted(counts.collect(), key = lambda x: x[1], reverse = 1)

# filter counts variable to only include tuples where the first element has a length longer than 3
bigCounts = [word for ix, word in enumerate(counts) if len(word[0]) > 3]

# print total number of unique words with length > 3
print('\nUnique words with length > 3: {}'.format(len(bigCounts)))

# print sum of utilization of words with length > 3
bigWordUsage = 0
for word in bigCounts:
        bigWordUsage += word[1]

print('\nUtilization of words with length > 3: {}'.format(bigWordUsage))

# print detailed result
print('\nDetailed results (top 50), descending order by individual word count: \n\t')
print(bigCounts[:50])
