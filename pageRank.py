import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import time

# Create spark context
conf = SparkConf()
sc = SparkContext(conf=conf)

# This loads the input file as an RDD, with each element being a string
# of the form "source destination" where source and destination
# are node id's representing the directed edge from node source
# to node destination. Note that the elements of this RDD are string
# types, hence you will need to map them to integers later.
lines = sc.textFile(sys.argv[1])

first = time.time()
### STUDENT PAGE RANK CODE START ###

beta = 0.8
iters = int(sys.argv[2])

# construct an RDD to represent the links matrix M
# parse the dinstinct "source destination" strings into (i,j) tuples
linkTuples = lines.distinct()\
                  .map(lambda x: (int(x.split()[0]), int(x.split()[1])))
# count the link tuples by key to find deg(i)
countsRDD = linkTuples.map(lambda (i,j): (i,1))\
                      .reduceByKey(lambda v1, v2: v1 + v2)\
                      .sortByKey(ascending=True)\
                      .map(lambda (i,d): d)
n = countsRDD.count()
counts = np.array(countsRDD.collect())

# initialize r
r = np.ones(n)/float(n)

for iteration in range(iters):
    for row in range(n):
        rowj  = np.zeros((1,n))
        rowj_sparse = np.array(linkTuples.filter(lambda (i, j): j == row).collect()).reshape((-1,2))
        
print(np.array(rowj_sparse).reshape((-1,2)))

     # this was for doing the matrix computation as an RDD
#    matrixProd = linkTuples.map(lambda (i, j): (i, r[j-1]))\
#                           .reduceByKey(lambda v1, v2: v1 + v2)\
#                           .map(lambda (i,sumR): (i, beta*(sumR / float((counts[i-1])) + (1 - beta) / float(n))))
#    r = np.array(matrixProd.sortByKey(ascending=True).map(lambda (i,j): j).collect())
    
sorted_idx = np.argsort(r)

top = r[sorted_idx[:5]]
bottom = r[sorted_idx[n-6:]]

print(counts[:5])
print('top:')
print(sorted_idx[0:5]+1)
print(top)
print('bottom')
print(sorted_idx[n-6:]+1)
print(bottom)

### STUDENT PAGE RANK CODE END   ###
last = time.time()
print("Total Program Time: " + str(last - first))

# Do not forget to stop the spark instance
sc.stop()

