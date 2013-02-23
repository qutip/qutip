import sys
import csv
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time

#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qutip-only",
                    help="only run qutip benchmarks",
                    action='store_true')
parser.add_argument("--run-profiler",
                    help="run profiler on qutip benchmarks",
                    action='store_true')
args = parser.parse_args()

if not args.qutip_only:
    # Call matlab benchmarks (folder must be in Matlab path!!!)
    if sys.platform=='darwin':
        sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
        sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
    else:
        sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
        sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)

# read in matlab results
matlab_version = csv.reader(open('matlab_version.csv'),dialect='excel')
matlab_times = genfromtxt('matlab_benchmarks.csv', delimiter=',')

# setup hardware and matlab info lists
platform=[hardware_info()]
for row in matlab_version:
    matlab_info=[{'version': row[0],'type': row[1]}]

qutip_info=[{'qutip':qutip.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__}]
#---------------------
#Run Python Benchmarks
#---------------------
def run_tests():
    #setup list for python times
    python_times=[]
    test_names=[]
    
    #test 1
    #Construct Jaynes-Cumming Hamiltonian with Nc=10, Na=2.
    test_names+=['Qobj add [20]']
    wc = 1.0 * 2 * pi  
    wa = 1.0 * 2 * pi
    g  = 0.05 * 2 * pi
    Nc=10
    tic=time()
    a=tensor(destroy(Nc),qeye(2))
    sm=tensor(qeye(Nc),sigmam())
    H=wc*a.dag()*a+wa*sm.dag()*sm+g*(a.dag()+a)*(sm.dag()+sm)
    toc=time()
    python_times+=[toc-tic]
    
    #test 2
    #tensor 6 spin operators.
    test_names+=['Qobj tensor [64]']
    tic=time()
    tensor(sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax())
    toc=time()
    python_times+=[toc-tic]
    
    #test 3
    #ptrace 6 spin operators.
    test_names+=['Qobj ptrace [64]']
    out=tensor([sigmax(),sigmay(),sigmaz(),sigmay(),sigmaz(),sigmax()])
    tic=time()
    ptrace(out,[1,3,4])
    toc=time()
    python_times+=[toc-tic]

    #test 4
    #test expm with displacement and squeezing operators.
    test_names+=['Qobj expm [20]']
    N=20
    alpha=2+2j
    sp=1.25j
    tic=time()
    coherent(N,alpha)
    squeez(N,sp)
    toc=time()
    python_times+=[toc-tic]

    #test 5
    #cavity+qubit steady state
    test_names+=['JC SS [20]']
    kappa=2;gamma=0.2;g=1;wc=0
    w0=0;N=10;E=0.5;wl=0
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

    #test 6
    #cavity+qubit master equation
    test_names+=['JC ME [20]']
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

    #test 7
    #cavity+qubit monte carlo equation
    test_names+=['JC MC [20]']
    kappa = 2; gamma = 0.2; g = 1;
    wc = 0; w0 = 0; wl = 0; E = 0.5;
    N = 10;
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
    
    #test 8
    #cavity+qubit wigner function
    test_names+=['Wigner [10]']
    kappa = 2; gamma = 0.2; g = 1;
    wc = 0; w0 = 0; wl = 0; E = 0.5;
    N = 10;
    tlist = linspace(0,10,200);
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
    out=mesolve(H, psi0, tlist, [C1, C2], [])
    rho_cavity=ptrace(out.states[-1],0)
    xvec=linspace(-10,10,200)
    tic=time()
    W=wigner(rho_cavity,xvec,xvec)
    toc=time()
    python_times+=[toc-tic]

    return python_times, test_names

if args.run_profiler:
    import cProfile
    cProfile.run('run_tests()', 'qutip_benchmarks_profiler')
    import pstats
    p = pstats.Stats('qutip_benchmarks_profiler')
    p.sort_stats('cumulative').print_stats(30)

else:
    python_times, test_names = run_tests()
    factors=matlab_times/array(python_times)
    data=[]
    for ii in range(len(test_names)):
        entry={'name':test_names[ii],'factor':factors[ii]}
        data.append(entry)

    f = open("benchmark_data.js", "w")
    f.write('data = ' + str(data) + ';\n')
    f.write('platform = ' + str(platform) + ';\n')
    f.write('qutip_info = ' + str(qutip_info) + ';\n')
    f.write('matlab_info= ' + str(matlab_info) + ';\n')
    f.close()