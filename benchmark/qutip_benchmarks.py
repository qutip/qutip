import sys
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time
#Call matlab benchmarks (folder must be in Matlab path!!!)
if sys.platform=='darwin':
    sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
else:
    sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)

#read in matlab results
matlab_times = genfromtxt('matlab_benchmarks.csv', delimiter=',')

#setup list for python times
python_times=[]
test_names=[]

#---------------------
#Run Python Benchmarks
#---------------------
num_tests=5

#Construct Jaynes-Cumming Hamiltonian with Nc=20, Na=2.
test_names+=['Qobj add']
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

#Construct Jaynes-Cumming Hamiltonian with Nc=20, Na=2.
test_names+=['Qobj expm']
N=25
alpha=2+2j
sp=1.25j
tic=time()
coherent(N,alpha)
squeez(N,sp)
toc=time()
python_times+=[toc-tic]

#cavity+qubit steady state
test_names+=['JC SS']
kappa=2;gamma=0.2;g=1;wc=0
w0=0;N=5;E=0.5;wl=0
tic=time()
ida=qeye(N)
idatom=qeye(2)
a=tensor(destroy(N),idatom)
sm=tensor(ida,sigmam())
H=(w0-wl)*sm.dag()*sm+(wc-wl)*a.dag()*a+1j*g*(a.dag()*sm-sm.dag()*a)+E*(a.dag()+a)
C1=sqrt(2*kappa)*a
C2=sqrt(gamma)*sm
C1dC1=C1.dag() * C1
C2dC2=C2.dag() * C2
L = liouvillian(H, [C1, C2])
rhoss=steady(L)
toc=time()
python_times+=[toc-tic]

#cavity+qubit master equation
test_names+=['JC ME']
kappa = 2; gamma = 0.2; g = 1;
wc = 0; w0 = 0; wl = 0; E = 0.5;
N = 10;
tlist = linspace(0,10,200);
tic=time()
ida    = qeye(N)
idatom = qeye(2)
a  = tensor(destroy(N),idatom)
sm = tensor(ida,sigmam())
H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)
C1=sqrt(2*kappa)*a
C2=sqrt(gamma)*sm
C1dC1=C1.dag()*C1
C2dC2=C2.dag()*C2
psi0 = tensor(basis(N,0),basis(2,1))
rho0 = psi0.dag() * psi0
mesolve(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a])
toc=time()
python_times+=[toc-tic]

#cavity+qubit monte carlo equation
test_names+=['JC MC']
kappa = 2; gamma = 0.2; g = 1;
wc = 0; w0 = 0; wl = 0; E = 0.5;
N = 4;
tlist = linspace(0,10,200);
tic=time()
ida = qeye(N)
idatom = qeye(2)
a  = tensor(destroy(N),idatom)
sm = tensor(ida,sigmam())
H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)
C1=sqrt(2*kappa)*a
C2=sqrt(gamma)*sm
C1dC1=C1.dag()*C1
C2dC2=C2.dag()*C2
psi0 = tensor(basis(N,0),basis(2,1))
mcsolve_f90(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a],options=Odeoptions(gui=False))
toc=time()
python_times+=[toc-tic]

#-- normalize times and get background colors using wigner_cmap
factors=np.round(matlab_times/array(python_times),2)

data=[]
for ii in range(num_tests):
    entry={'name':test_names[ii],'factor':factors[ii]}
    data.append(entry)

f = open("benchmark_data.js", "w")
f.write('data = ' + str(data) + ';')
f.close()