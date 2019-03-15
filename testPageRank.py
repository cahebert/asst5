import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import time

# Create spark context
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
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

# parse the dinstinct "source destination" strings into (i, [j]) where [j] is the list of all the nodes i points to
linkTuples = lines.distinct()\
                  .map(lambda x: (int(x.split()[0]), int(x.split()[1])))\
                  .groupByKey()
#                  .persist()

n = linkTuples.count()
#initialize rank vector
ranks = linkTuples.map(lambda x: (x[0], 1 / float(n)))

def mapping(jList, r):
    nNeighbors = len(jList)
    for j in jList:
        # jList is all the nodes that i points to (i.e. deg(i))
        yield (j, r / nNeighbors) 


for iteration in range(iters):
    # after join you have (i, (jList, r(i)))
    contribution = linkTuples.join(ranks)
    contribution2 = contribution.flatMap(lambda i_jList_r: mapping(i_jList_r[1][0], i_jList_r[1][1]))

    ranks = contribution2.reduceByKey(lambda r1, r2: r1 + r2)\
                         .mapValues(lambda sumR: sumR * beta + (1 - beta) / float(n))


r_bottom = np.array(ranks.takeOrdered(5, key = lambda x: x[1]))
r_top = np.array(ranks.takeOrdered(5, key = lambda x: -x[1]))

print('top five')
print(r_top)

print('bottom five')
print(r_bottom)

####
#print(r[idx_sort[:5],0])
#print(r[idx_sort[:5],1])

#print(r[idx_sort[n-6:],0])
#print(r[idx_sort[n-6:],1][::-1])

### STUDENT PAGE RANK CODE END   ###
last = time.time()
print("Total Program Time: " + str(last - first))

# Do not forget to stop the spark instance
sc.stop()

