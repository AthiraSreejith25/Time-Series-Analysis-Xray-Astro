
import math
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy.stats
from scipy.stats import poisson

#random.seed(0)
#np.random.seed(0)
"""
def Time_Series(rate,n):

    c = 0
    k = 1 #the bin number
    int_arrival_time = 0
    times = []
    occ_time = 0
    occurences = []
    int_counts = np.zeros(100)
    
    for i in range(n):
        
        #int_arrival_time =  data_expon = scipy.stats.expon.rvs(scale=40,loc=0,size=100)
        int_arrival_time = -math.log(1.0 - random.random()) / rate
        times.append(int_arrival_time)
        occ_time += int_arrival_time
        occurences.append(occ_time)
        
        #c += 1


            
    
                
    #print(occurences)
    return(occurences)
    #return -math.log(1.0 - random.random()) / rateParameter



start = 0
step = 5
counts = []
count = np.zeros(step)
#bs = 5
l = []
lambdai = step*5/100
T = Time_Series(lambdai,1000)
#T = scipy.stats.geom.rvs(0.77, size=100)
print(T)

    #step += 5
noi = int(T[-1])//step
counts = np.zeros(noi + 1 , dtype = int)
for k in T:
    counts[int(k//step)] += 1

    #count = np.zeros(step)
#print(counts)
    #for ind in range(len(count)):
count = np.bincount(counts)
#print(count)
li = len(counts)
print(statistics.mean(counts) , 1/lambdai)
print(scipy.stats.kstest(count, 'poisson',args = (count,statistics.mean(counts))))
print(scipy.stats.kstest(count, 'poisson',args = (count,1/lambdai)))

"""



def arrivalt(rate,timeint,n):#returns counts
    time = 0
    arrt = []
    
    intar = scipy.stats.expon.rvs(scale=1/rate*timeint,loc=0,size=n)

    for i in intar:
        time += i
        #print(time)
        arrt.append(time)

    noi = int(arrt[-1])//timeint
    counts = np.zeros(noi + 1 , dtype = int)
    for k in arrt:
        counts[int(k//timeint)] += 1
    #print(counts)
    #mean = statistics.mean(counts)
    count = np.bincount(counts)
    #print(count)
    
    return(count,counts)


obl,oo = arrivalt(0.05,1000,1000000)

print(oo)
#mean = statistics.mean(oo)
mean = sum(oo)/len(oo)
tree = 0
obi = []
for i in obl:
    tree += i
    obi.append(tree)
obl = [i/len(oo) for i in obl]
obi = [tree/i for i in obi]#experimental cdf
print(obi)
print(scipy.stats.kstest(obl, poisson.rvs(mean,size = len(obl))))
print(scipy.stats.kstest(obl ,'poisson', args =(mean,len(obl)) ))
print(obl)

#plt.show()
plt.plot(np.arange(0,len(obl)) , obl , 'g')
plt.plot(np.arange(0,len(obl)) , [(2.7182**(-mean))*((mean)**i )/math.factorial(i) for i in range(len(obl))] , 'r')

plt.show()
