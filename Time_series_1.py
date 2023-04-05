
import math
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from timeit import default_timer as timer
import scipy.stats



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
step = 10
counts = []
count = np.zeros(step)
#bs = 5
l = []
lambdai = 40/100
for klikli in range(20):
    random.seed(0)
    #lambdai += 0.25
    T = Time_Series(lambdai,100)
    print(T)

    #step += 5
    noi = int(T[-1])//step
    counts = np.zeros(noi + 1 , dtype = int)

    for k in T:
        counts[int(k//step)] += 1

    #count = np.zeros(step)
    print(counts)
    #for ind in range(len(count)):
    count = np.bincount(counts)


    av = statistics.mean(counts)
    #print(av)
    #print(statistics.variance(counts))
    #print(skew(counts))
    #print(counts, len(counts),count, counts.count(1))
    #print(Moments(count))
    #print(statistics.stdev(counts))
    #print(kurtosis(counts) , 1/av , 1/(statistics.variance(counts))) 
    #print(1/(kurtosis(counts)))
    #print(skew(counts),(av)**(-1/2)) 
    #print((skew(counts))**(-2))
    #print(Poisson(Time_Series(1/50,100)))

    x = []
    y = []
    print(count)
    for i in range(len(count)):
        if count[i] != 0:
            a = math.log((math.factorial(i))*count[i]/len(counts))
            y.append(a)
            x.append(i)

    #print(x,y)                        
    plt.plot(x , y , 'g')
    #plt.savefig("Poisson_distribution_with_binsize_{}.png".format(step))
    #plt.show()
    plt.close()



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
    #print( "m =",m)
    #print( "c =",c)
    # error calculation

    sigy2 = 0.0
    for i in range(n):
        sigy2 = sigy2 + (y[i] - m*x[i] - c)*(y[i] - m*x[i] - c)
    # loop over
    sigy = math.sqrt(sigy2/float(n-2))
    sigm = sigy * math.sqrt(float(n)/delta)
    sigc = sigy * sx * math.sqrt(1.0/delta)
    #print ('error in slope = ', sigm)
    l.append(sigm/m)
    #print ('error in intercept = ', sigc)
    #print("relative error = " , sigc/c)
    #print("expected slope = {}".format(math.log(25/80)))                        

print(statistics.mean(l))
print(max(l))
print(min(l))
print(l)


"""    
    stop = int(max(T)) + 1
    yu = 0
    for j in range(start,stop,step):
            
        for t in T:
            if t < j + step and t >= j:
                    #print(t)
                yu += 1
            #countie += yu
        counts.append(yu)
        count[yu] += 1
        yu = 0
    #print(count)
    #mean = counts.count(1) / len(counts)
"""
"""
T = Time_Series(1/80,100)
#start = 0
#step = 25
stop = int(max(T)) + 1
yu = 0

#new---------------
st2 = timer()
noi = int(T[-1])//step
counts = np.zeros(noi + 1 , dtype = int)

for k in T:
    counts[int(k//step)] += 1

#count = np.zeros(step)

#for ind in range(len(count)):
count = np.bincount(counts)
#new------------    

st1 = timer()
print(count,counts)
print(st1 - st2)
"""
"""
#slower----------
st3 = timer()
countie = np.arange(0,(int(T[-1])//step)*step,step)
print(countie)
countrie = np.digitize(T,countie)
print(countrie)
count = np.bincount(countrie)

st4 = timer()
print(count,countrie)
print(st4 - st3)
#slower----------
"""
"""
counts = []
st1 = timer()
#number of intervals
for j in range(start,stop,step):
        
    for t in T:
        if t < j + step and t >= j:
                #print(t)
            yu += 1
        #countie += yu
    counts.append(yu)
    count[yu] += 1
    yu = 0
print(counts, count)
st2 = timer()
print(st2 - st1)
#mean = counts.count(1) / len(counts)
"""
"""
av = statistics.mean(counts)
print(av)
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
plt.show()




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

"""








#Junk-----------------

"""
def Poisson(times):
    start = 0
    bin_size = 10
    stop = start + bin_size
    counts = np.zeros(100)
    yu = 1
    for i in range(len(times)):

        if times[i] < stop:
            yu += 1
            print(yu)
            print(times[i],stop)

        else:
            print(yu)
            counts[yu - 1] += 1
            yu = 1

            counts[0] += (times[i] - stop)//bin_size
            print((times[i] - stop)//bin_size , times[i] , stop)
            stop += ((times[i] - stop)//bin_size)*bin_size
            

        
    counts[yu] += 1
    
    
            

    return(counts)
"""
"""
def Moments(freq_dist):
    #m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    mean = 0
    total = 0

    for i in range(len(freq_dist)):
        mean += i*(freq_dist[i])
        total += freq_dist[i]
        

    mean = mean/total
    print(mean)
    for i in range(len(freq_dist)):
        #m1 += i*(freq_dist[i] - mean)
        m2 += i*(freq_dist[i] - mean)**2
        m3 += i*(freq_dist[i] - mean)**3
        m4 += i*(freq_dist[i] - mean)**4
        


    return(m2/total, m3/(m2/total)**(3/2) , m4/(m2/total)**2)

"""

    
"""
#rateParameter = 0.5
countie = np.zeros(5)
counts = []
start = 0

#print(stop)
step = 1
yu = 0
"""
#print(Time_Series(1/50,10,3))
#T = Time_Series(1/50,100)
"""
stop = int(max(T)) + 1
print(stop)

for j in range(start,stop,step):

    for t in T:
        if t < j + step and t >= j:
            #print(t)
            yu += 1
    #countie += yu
    counts.append(yu)
    countie[yu] += 1
    yu = 0

print(countie)
print(sum(counts))
"""
"""
countie =np.zeros(45)
for t in range(len(counts)):

    countie[t] += 1

print(countie)        
"""
"""


#works
tie = 0 
for ttt in range(1000):
    T = Time_Series(1/80,10)

    stop = int(max(T)) + 1

    for j in range(start,stop,step):
        
        for t in T:
            if t < j + step and t >= j:
                #print(t)
                yu += 1
        #countie += yu
        counts.append(yu)
        yu = 0

    tie += sum(counts)/len(counts)

print(tie/1000)
"""


"""
#start =
bin_size =
stop = start + bin_size
while stop < int(max(freq)) + 1 :
"""





        
"""

time_start = 0
    bin_size = 1
    count_per_bin = []

    c = 0
    
if occ_time < time_start + bin_size:
            #count_per_bin.append(c-1)
            c += 1

        else:

            zero_occ_bins = int((occ_time - time_start)//bin_size) - 1 
            for i in range(zero_occ_bins):

                count_per_bin.append(0)

            count_per_bin.append(c-1)
            
            time_start += bin_size*zero_occ_bins
            c = 1

"""


"""
        if occ_time < k*bin_size:

            c += 1
            
            #print(occ_time,k*bin_size)

        #elif i == n - 1:
            #int_counts[c] += 1
  
        else:

            if c != 0:
                print(occ_time)
            int_counts[c] += 1
            c = 1

            int_counts[0] += (occ_time - (k*bin_size))//bin_size
            k += 1 + (occ_time - (k*bin_size))//bin_size
                        
"""
"""
import numpy as np
import random
#print(random.randint(0,0))
#print(np.random.randint(0,0))
size = 5
Ag_id = 4
Ag = [(0,1,2),(0,2,3)]
surf = np.zeros((2,size,size),dtype = int)
surf[0][1][2] = Ag_id
surf[0][2][3] = Ag_id
def Ag_check():

        for mol in Ag:

                if surf[mol[0] , mol[1] , mol[2]] != Ag_id:
                        return(False)

        return(True)
                        
print(Ag_check())
"""
"""
st5 = timer()
def binn(data,low,size):

    freq = []
    classmarks = []
    dat = data[:]
    
    k = 0

    while low+((k+2)*size) < max(data):
        count = 0
        for i in dat:

            if low+(k*size) < i <= low+((k+1)*size):

                count += 1
                dat.remove(i)

        freq.append(count)
        classmarks.append(low+(k*size))
        k += 1

    return (classmarks,freq)
print(binn(T,0,25))
st6 = timer()
print(st6 - st5)
"""
