# imports and tools
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as f

from pyspark.ml.feature import CountVectorizer

import re
import os
import numpy as np
import operator

import matplotlib.pyplot as plt

# create SparkContext object
spark = SparkSession.builder.appName("Assignment_03").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

####################################################################################
## part 1
# load data to dataframe
print('*' * 100)
print('Part 1 - load data into dataframe\n')
path = 's3://tdp-ml-datasets/misc/Amazon.csv'
dataAmazon = spark.read.load(path , format = 'csv', header = 'true', inferschema = 'true', sep = ",")

dataAmazonRDD = dataAmazon.rdd
dataAmazonRDD = dataAmazonRDD.map(lambda x: (x[0], list(x[1:])))

print('*' * 25)
print('\nAmazon\n')
for x in dataAmazonRDD.take(5):
    print(x)

path = 's3://tdp-ml-datasets/misc/Google.csv'
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

path = 's3://tdp-ml-datasets/misc/stopwords.txt' 
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
print('Part 3 - Return term frequency of a list of tokens\n')

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
googleTermFreq = dataGoogleTokens.map(lambda x: (x[0], termFreq(x[1], normalize = False)))
googleTermFreqNorm = dataGoogleTokens.map(lambda x: (x[0], termFreq(x[1], normalize = True)))

print('\nNot normalized')
for x in googleTermFreq.take(5):
    print(x)
print('\nNormalized')
for x in googleTermFreqNorm.take(5):
    print(x)


print('\nAmazon term frequency sample')
amazonTermFreq = dataAmazonTokens.map(lambda x: (x[0], termFreq(x[1], normalize = False)))
amazonTermFreqNorm = dataAmazonTokens.map(lambda x: (x[0], termFreq(x[1], normalize = True)))

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
print('Part 5 - Calculate IDFs and visualize lowest values \n')


def calcIdf(corpus):
    # get document count
    docCount = corpusDf.agg(f.countDistinct('id'))
    docCount = docCount.collect()[0][0]

    # explode token vector
    corpusDfExplode = (corpusDf.select('id','tokens',(f.explode('tokens').alias('indvToken'))))

    # count number of IDs that include each word to get document frequency
    docFreqs = corpusDfExplode.groupBy('indvToken').agg(f.countDistinct('id').alias('df')) 
    docFreqs = docFreqs.sort(f.desc('df'))
    
    idfDf = docFreqs.withColumn('idf', docFreqs.df / docCount)
    idfDf = docFreqs.withColumn('idf', f.log((docCount + 1) / (docFreqs.df + 1)))

    idfRdd = idfDf.select('indvToken','idf').rdd 
    return idfRdd 

idfRdd = calcIdf(corpus = corpusDf)
print('Five lowest IDF values')
print(idfRdd.take(5))

idfSubset = idfRdd.takeOrdered(10, key = lambda x: x[1])  
idfSubset = [x[1] for x in idfSubset]
plt.hist(idfSubset, bins = 5) 
plt.xlabel('token')
plt.ylabel('IDF')
plt.savefig('LowestIDF.png')

####################################################################################
## part 6
# 
print('*' * 100)
print('Part 6 - Calculate term frequencies and TF-IDF \n')

# 
print('Part A - Calculate term frequencies')
combinedTermFreq = amazonTermFreq.union(googleTermFreq)

def tfidfFunc(corpus, idfs):
    
    # transform input corpus and IDFs into dictionaries
    idfs = idfs.collectAsMap()
    keyTFs = corpus.collectAsMap()  
    
    # for each document in the corpus
    for key in keyTFs.keys():
        
        # for each token in the document
        for token in keyTFs[key].keys():

            # multiply the token's frequency by that term's IDF value
            keyTFs[key][token] = keyTFs[key][token] * idfs[token]
            
    return keyTFs
    
idfDict = tfidfFunc(combinedTermFreq, idfRdd)

print('Print first five items in idfDict, which is a dictionary where the key is the document ID and the value is an embedded dictionary containing the tokens and the corresponding TF-IDF values')
idfDictFirstFive = {k: idfDict[k] for k in list(idfDict)[:5]}  

for k in idfDictFirstFive.keys():
    print(idfDictFirstFive[k])
    print()
