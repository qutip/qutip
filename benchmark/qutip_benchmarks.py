import sys
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
from time import time
from pylab import *

#Call matlab benchmarks (folder must be in Matlab path!!!)
if sys.platform=='darwin':
    sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r matlab_benchmarks",shell=True)
else:
    sproc.call("matlab -nodesktop -nosplash -r matlab_benchmarks",shell=True)

#read in matlab results
matlab_times = genfromtxt('matlab_benchmarks.csv', delimiter=',')

#setup list for python times
python_times=[]
test_names=[]
#---------------------
#Run Python Benchmarks
#---------------------
num_tests=2

#Construct Jaynes-Cumming Hamiltonian with Nc=20, Na=2.
wc = 1.0 * 2 * pi  
wa = 1.0 * 2 * pi
g  = 0.05 * 2 * pi
Nc=20
tic=time()
a=tensor(destroy(Nc),qeye(2))
sm=tensor(qeye(Nc),sigmam())
H=wc*a.dag()*a+wa*sm.dag()*sm+g*(a.dag()+a)*(sm.dag()+sm)
toc=time()
python_times+=[toc-tic]
test_names+=['JC-model']

#Construct Jaynes-Cumming Hamiltonian with Nc=20, Na=2.
N=25
alpha=2+2j
sp=1.25j
tic=time()
coherent(N,alpha)
squeez(N,sp)
toc=time()
python_times+=[toc-tic]
test_names+=['Operator expm']

barh(arange(num_tests),matlab_times/array(python_times),align='center',height=0.5)
for ii in range(num_tests):
    text(0.5,ii,test_names[ii],color='w',fontsize=16,verticalalignment='center')
frame = gca()
for y in frame.axes.get_yticklabels():
    y.set_fontsize(0.0)
    y.set_visible(False)
for x in frame.axes.get_xticklabels():
    x.set_fontsize(12)
for tick in frame.axes.get_yticklines():
    tick.set_visible(False)
xlabel("Times faster than the Quantum Optics Toolbox",fontsize=16)
title('QuTiP vs. Quantum Optics Toolbox Performance')
show()

