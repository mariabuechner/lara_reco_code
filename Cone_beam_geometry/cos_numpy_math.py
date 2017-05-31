'''
Comparison of speed by using cos function from two different packages:
numpy and math.

'''
from math import cos 
from time import time
num_repeats = 10000000

start_time = time()
for i in range(num_repeats):
    cos(0.785568912)
print "math.cos() Elapsed time: " + str(time()-start_time)

from numpy import cos
start_time = time()
for i in range(num_repeats):
    cos(0.785568912)
print "numpy.cos() Elapsed time: " + str(time()-start_time)