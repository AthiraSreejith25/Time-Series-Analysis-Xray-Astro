
import math


"""
t = range(21)
y = []
#a = 1
tp = 20/5
#w = 2*math.pi/tp
#sin(2*math.pi/tp) = sin(2*math.pi)
#print(w)
for i in t:
    y.append(math.sin((2*math.pi/tp)*i))
print(y)
def ift(x):
    l = []
    for i in range(len(x)-1):
        if x[i]>x[i-1] and x[i]>x[i+1]:
            l.append(i)
    if x[19]>x[18] and x[19]>x[20]:
        l.append(19)
    return(l,len(l))
        
print(ift(y))

"""
"""
print(math.sin(6*math.pi/20*20))
print(math.sin(2*math.pi))
print(math.sin(4*math.pi))
"""

import numpy as np
import matplotlib.pyplot as plt
"""
# Make a quick sin plot
x = np.linspace( 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Save it in png and svg formats
save("signal", ext="png", close=False, verbose=True)
save("signal", ext="svg", close=True, verbose=True)

plt.plot([1, 2, 3])
plt.savefig('myfig.png')
"""
#plotting random distri
from scipy.stats import expon
data_expon = expon.rvs(scale=1,loc=0,size=1000)
#scale = 1/lambda

from scipy.stats import poisson
data_poisson = poisson.rvs(mu=3, size=10000)

from scipy.stats import binom
data_binom = binom.rvs(n=10,p=0.8,size=10000)

from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000,p=0.6)
#p = probability of success


#geometric
from scipy.stats import geom
r = geom.rvs(p, size=1000)

from scipy.stats import hypergeom
R = hypergeom.rvs(M, n, N, size=10)
# m= total size , n = number of elements with a given charecteristic(special)  N = number sampled, R = number of special ones among sampled

R = zipf .pmf(a, b)

#negative binomial
R = nbinom .rvs(a, b, size = 10)

"""
#multiplots into pdf-------------

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # if LaTeX is not installed or error caught, change to `usetex=False`
    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
                                       # attach metadata to a page
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x ** 2, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = 'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()


"""
