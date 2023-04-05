    
import math
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from timeit import default_timer as timer
import scipy.stats

np.random.seed(0)
timestep = 10
#int_arrival_time = scipy.stats.expon.rvs(scale=10*timestep,loc=0,size=1000)
#int_arrival_time  = scipy.stats.binom.rvs(n=10,p=0.08,size=100)

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


#print(int_arrival_time)
def occ_times(lis):

    occurences = [lis[0]]

    for i in range(1,len(lis)):

        occurences.append(lis[i] + occurences[i-1])

    return(occurences)

#print(occ_times(int_arrival_time))
#T = scipy.stats.geom.rvs(p = 0.09, size=100)
M = 1000
n = 30
N = 50
T = scipy.stats.hypergeom.rvs(M, n, N, size=10)
print(T)
#T = occ_times(int_arrival_time)
#lambdai = 80/(100*timestep)
lambdai = 40/(100*timestep)
#T = Time_Series(lambdai*timestep,100) - fr poisson

start = 0
step = timestep
#stop = 1 +(int(T[-1])//step) - fr poisson
stop = 1 + max(T)

counts = np.zeros(stop , dtype = int)

for k in T:
        counts[int(k//step)] += 1

count = np.bincount(counts)

x = []
y = []
print(count)
for i in range(len(count)):
    if count[i] != 0:
        a = math.log((math.factorial(i))*count[i]/len(counts))
        y.append(a)
        x.append(i)

    #print(x,y)                        
ari = 0.09
plt.plot(x , y , 'g')
plt.savefig("4Geometric_with_{}.png".format(ari))
plt.show()


av = statistics.mean(counts[1:])
print("averagew",av)
print(statistics.variance(counts))
#print(skew(counts))
#print(counts, len(counts),count, counts.count(1))
#print(Moments(count))
print(statistics.stdev(counts))
print(kurtosis(counts) , 1/av , 1/(statistics.variance(counts))) 
print(1/(kurtosis(counts)))
print(skew(counts),(av)**(-1/2)) 
print((skew(counts))**(-2))
#print(Poisson(Time_Series(1/50,100)))

x = []
y = []
for i in range(len(count)):
    if count[i] != 0:
        a = math.log((math.factorial(i))*count[i]/len(counts))
        y.append(a)
        x.append(i)

print(x,y)                        
plt.plot(x , y , 'g')
#plt.show()




n= len(x)
# Data
#x = [1.0,2.0,3.0,4.0,5.0]
#y = [2.0,4.0,6.0,8.0,10.0]
sx = 0.0
sx2 = 0.0
sy = 0.0
sxy = 0.0
for i in range(n):
    sx = sx + x[i]
    sx2 = sx2 + x[i]*x[i]
    sy = sy + y[i]
    sxy = sxy + x[i]*y[i]
# loop over
delta = - sx*sx + float(n)*sx2
m = -(sx*sy-float(n)*sxy)/delta
c = -(sx*sxy - sx2*sy)/delta
print( "m =",m)
print( "c =",c)
# error calculation

sigy2 = 0.0
for i in range(n):
    sigy2 = sigy2 + (y[i] - m*x[i] - c)*(y[i] - m*x[i] - c)
# loop over
sigy = math.sqrt(sigy2/float(n-2))
sigm = sigy * math.sqrt(float(n)/delta)
sigc = sigy * sx * math.sqrt(1.0/delta)
print ('error in slope = ', sigm)
print("relative error = " , sigm/m)
print ('error in intercept = ', sigc)
print("relative error = " , sigc/c)
print("expected slope = {}".format(math.log(25/80)))                        


