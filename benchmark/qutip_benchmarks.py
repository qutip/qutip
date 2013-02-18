import sys, HTML
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time
import matplotlib as mpl
from matplotlib import pyplot, cm
from pylab import *
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
num_tests=4

#Construct Jaynes-Cumming Hamiltonian with Nc=20, Na=2.
test_names+=['Build JC Hamiltonian']
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
test_names+=['Operator expm']
N=25
alpha=2+2j
sp=1.25j
tic=time()
coherent(N,alpha)
squeez(N,sp)
toc=time()
python_times+=[toc-tic]

#cavity+qubit steady state
test_names+=['cavity+qubit steady state']
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
test_names+=['cavity+qubit master equation']
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


#-- normalize times and get background colors using wigner_cmap
normed_times=np.round(matlab_times/array(python_times),1)
min_time=min(normed_times)
max_time=max(normed_times)
cmap=wigner_cmap(normed_times-1)


#Build HTML page for results
col_styles=[]
for kk in normed_times:
    level=kk/(max_time-min_time)
    color=np.round(array(cmap(level)[0:3])*255,1)
    if kk-1>0:
        col_styles+=["height:50px; width:75px;font-size: large;color: #E4D00A;background-color:rgb(%d,%d,%d)" % (color[0],color[1],color[2])]
    else:
        col_styles+=["height:50px;width:75px;font-size: large;color: #00FF00;background-color:rgb(%d,%d,%d)" % (color[0],color[1],color[2])]
html_file= 'qutip_vs_matlab_benchmarks.html'
f = open(html_file, 'w')
htmlcode = HTML.table([list(normed_times)],
                    header_row =test_names,
                    col_align=['center']*num_tests,
                    col_styles=col_styles)

f.write(htmlcode + '<p>\n')
f.close()
