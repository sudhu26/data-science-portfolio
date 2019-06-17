

import numpy as np
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors

# create SparkContext object
spark = SparkSession.builder.appName("Unit03_IntroML").getOrCreate()
sc = spark.sparkContext

# Spark MLlib supports local vectors and matrices stores on a single machine, as well as distributed arrangements 
# of these objects.

## local vector
# MLlib supports two types of local vectors - dense and spark. For example, (1.0, 0.0, 3.0) can be represented
# in a dense format as [1.0, 0.0, 3.0], and also as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector
# [0, 2] represents the indices of the non-zero values, and [1.0, 3.0] are the non-zero values occurring at those
# indices

# numpy arrays are dense vectors
dv1 = np.array([1.0, 0.0, 3.0])
print(dv1)

# lists are dense vectors
dv2 = [1.0, 0.0, 3.0]
print(dv2)

# create a SparseVector
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])
print(sv1)

# use single column scipy csc_matrix as a sparse vector
sv2 = sps.csc_matrix((np.array([1.0, 3.0]), np.array([0, 2]), np.array([0, 2])), shape = (3, 1))
print(sv2)

## labeled point
# a labeled point is a local vector (dense or sparse) that is associated with a response. These are the labels 
# utilized by supervised learning algorithms.

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

# create labeled point with a positive label and a dense feature vector
pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
print(pos)

# create labeled point with a negative label and a sparse feature vector
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 0.3]))
print(neg)

## Sparse training data
# MLlib supports reading training examples where each line represents a labeled sparse feature vector

from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import os


# print(os.environ)
examples = MLUtils.loadLibSVMFile(sc, '{}/data/mllib/sample_libsvm_data.txt'.format(os.environ['SPARK_HOME']))
print(examples.take(5))


## distributed matrix
# a distributed matrix is stored across one or more RDDs. There are three distributed matrices implemented by MLlib.
# RowMatrix, IndexedRowMatrix, CoordinateMatrix

# A RowMatrix is a row-oriented distributed matrix without meaningful row indices. An example would be a training
# dataset with a series of observations as rows and feature vector as columns. 
from pyspark.mllib.linalg.distributed import RowMatrix

# a RDD of local vectors
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

mat = RowMatrix(rows)

m = mat.numRows()
n = mat.numCols()

print(m)
print(n)

# An IndexedRowMatrix is similar to a RowMatrix but has row indices, which can be used to identify specific rows, 
# which is useful for executing join.

from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

# a RDD of indexed rows
indexed = sc.parallelize([IndexedRow(0, [1, 2, 3])
                        ,IndexedRow(1, [4, 5, 6])
                        ,IndexedRow(2, [7, 8, 9])
                        ,IndexedRow(3, [10, 11, 12])
    ])
mat = IndexedRowMatrix(indexed)
print(mat)

# convert to row matrix
rowMat = mat.toRowMatrix()
print(rowMat)

# A CoordinateMatrix is distributed and stored in an object called a coordinate list.

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

entries = sc.parallelize([MatrixEntry(0, 0, 1.2)
                        ,MatrixEntry(1, 0, 2.1)
                        ,MatrixEntry(6, 1, 3.7)
    ])
mat = CoordinateMatrix(entries)

m = mat.numRows()
n = mat.numCols()

print(m)
print(n)

# convert to indexed row matrix
rowMat = mat.toIndexedRowMatrix()
print(rowMat)
