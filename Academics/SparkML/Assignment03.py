# imports and tools
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.mllib.feature import HashingTF, IDF

from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline

import re
import os
import numpy as np

import matplotlib.pyplot as plt

# create SparkContext object
spark = SparkSession.builder.appName("Assignment_03").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

#https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4574377819293972/2261876665858806/3186223000943570/latest.html
####################################################################################
## part 1
# load data to dataframe
print('*' * 100)
print('Part 1 - load data into dataframe\n')
path = '../../data/Amazon.csv'
dataAmazon = spark.read.load(path , format = 'csv', header = 'true', inferschema = 'true', sep = ",")

dataAmazonRDD = dataAmazon.rdd
dataAmazonRDD = dataAmazonRDD.map(lambda x: (x[0], list(x[1:])))

print('*' * 25)
print('\nAmazon\n')
for x in dataAmazonRDD.take(5):
    print(x)

path = '../../data/Google.csv'
dataGoogle = spark.read.load(path , format = 'csv', header = 'true', inferschema = 'true', sep = ",")

dataGoogleRDD = dataGoogle.rdd
dataGoogleRDD = dataGoogleRDD.map(lambda x: (x[0], list(x[1:])))

print('*' * 25)
print('\nGoogle\n')
for x in dataGoogleRDD.take(5):
    print(x)

####################################################################################
## part 2
# 
print('*' * 100)
print('Part 2 - Bag-of-words\n')

print('*' * 50)
print('Part 2a & 2b- implement function to return non-empty string and remove stopwords\n')

path = '../../data/stopwords.txt' 
stopwords = sc.textFile(path).collect()

quickbrownfox = 'A quick brown fox jumps over the lazy dog.'
regexRule = '\W+'
def tokenizer(string):
    # parse string and lower case
    string = re.split(regexRule, string)
    string = filter(None, string)
    string = [wrd.lower() for wrd in string]
    
    # remove stopwords
    cleanString = []
    for wrd in string:
        if not (wrd in stopwords):
            cleanString.append(wrd)
    return cleanString
    return string

sent = 'the skyscraper is very tall and slightly gray'
print('original sentence:\n{}'.format(sent))
print('processed sentence:\n{}'.format(tokenizer(sent)))

print('*' * 50)
print('Part 2c- tokenize Amazon and Google datasets\n')

print('*' * 25)
print('\nAmazon\n')

# tokenize product title
dataAmazonTokens = dataAmazonRDD.map(lambda x: (x[0], tokenizer(x[1][0])))
for x in dataAmazonTokens.take(5):
    print(x)

print('*' * 25)
print('\nGoogle\n')

# tokenize product title
dataGoogleTokens = dataGoogleRDD.map(lambda x: (x[0], tokenizer(x[1][0])))
for x in dataGoogleTokens.take(5):
    print(x)

####################################################################################
## part 3
# 
print('*' * 100)
print('Part 3 - Return term frequency of a list of tokens using MLLIB TF functions\n')

def termFreq(tokens, normalize = False):
    # capture term frequency for input token list
    termFreqDict = {}
    for token in tokens:
        if token in termFreqDict:
            termFreqDict[token] += 1
        else:
            termFreqDict[token] = 1

    # normalize counts based on number of tokens
    if normalize:
        for token in termFreqDict:
            termFreqDict[token] /= float(len(tokens))
    return termFreqDict



print('Google term frequency sample')
googleTermFreq = dataGoogleTokens.map(lambda x: x[1]).map(termFreq)
googleTermFreqNorm = dataGoogleTokens.map(lambda x: termFreq(x[1], normalize = True))

print('\nNot normalized')
for x in googleTermFreq.take(5):
    print(x)
print('\nNormalized')
for x in googleTermFreqNorm.take(5):
    print(x)


print('\nAmazon term frequency sample')
amazonTermFreq = dataAmazonTokens.map(lambda x: x[1]).map(termFreq)
amazonTermFreqNorm = dataAmazonTokens.map(lambda x: termFreq(x[1], normalize = True))

print('\nNot normalized')
for x in amazonTermFreq.take(5):
    print(x)
print('\nNormalized')
for x in amazonTermFreqNorm.take(5):
    print(x)


####################################################################################
## part 4
# 
print('*' * 100)
print('Part 4 - Combine Amazon and Google RDDs\n')

corpus = dataAmazonTokens.union(dataGoogleTokens)

print('Print every 100th key/value pair in corpus')
for x in corpus.collect()[::100]:
    print(x)


corpusDf = corpus.toDF(['id','tokens'])

####################################################################################
## part 5
# 
print('*' * 100)
print('Part 5 - \n')

vectorizer = CountVectorizer(inputCol = 'tokens', outputCol = 'cv').fit(corpusDf)
vectorized = vectorizer.transform(corpusDf)

idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5).fit(vectorized)
idfed = idf.transform(vectorized)

####################################################################################
## part 6
# 
print('*' * 100)
print('Part 6 - \n')

