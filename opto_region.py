"""
Requires QuTiP version 2.3 or higher. www.qutip.org

This QuTiP script generates the data presented in Fig. 1(a-d) from:

P. D. Nation, "Nonclassical Mechanical States in a Optomechanical 
Micromaser Analogue".  

This script requires at least 32Gb of memory 
per CPU.  The number of CPU's can be changed using the 'num_cpus' 
keyword argument on line #100. For a single CPU @ 2.4Ghz, this script
will take ~35 days to finish.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
from scipy import *
import numpy as np
from qutip import *

Nc = 5						 # Number of cavity states
Nm = 10                     # Number of mech. states
E = 0.05				     # Coherent state amplitude
kappa = 0.05			     # Cavity damping rate
Qm = 1e4                     # Mech quality factor
gamma = 1.0/Qm			     # Mech damping rate
n_th = 31                    # Mech bath temperature
xvec = linspace(-25,25,1596) # Range to evalue Wigner func.

# Define grid parameters
dw_num = 29
dg_num = 31
G0 = linspace(0.5,8,dg_num)*kappa
DW = linspace(-8.0,0,dw_num)



def ss_iter(delta):
    """
    Calculates the steady state at a given detuning
    value for a fixed g0.
    """
    H = -delta*(a.dag()*a)+H1
    rho_ss = steadystate(H, c_ops, method='direct')
    rho_cav = ptrace(rho_ss,0) # partial trace to get cavity DM
    numa = expect(num_a,rho_cav)
    cav_comm = expect(comm_a,rho_cav)
    cav_var = variance(num_a,rho_cav)
    rho_mech = ptrace(rho_ss,1) # partial trace to get mech. DM
    numb = expect(num_b,rho_mech)
    fano = variance(num_b,rho_mech)/numb
    mech_comm = expect(comm_b,rho_mech)
    W,yvec = wigner(rho_mech,xvec,xvec,method='fft') # Wigner function
    dxdy = (xvec[1]-xvec[0])*(yvec[1]-yvec[0])
    neg_inds = where(W<0.0)
    neg_weight = np.sum(np.abs(W[neg_inds[0],neg_inds[1]])*dxdy)
    pos_inds = where(W>0.0)
    pos_weight = np.sum(np.abs(W[pos_inds[0],pos_inds[1]])*dxdy)
    return numa,numb,pos_weight,neg_weight,fano,mech_comm,cav_comm,cav_var

#operators
idc = qeye(Nc)
idm = qeye(Nm)
a = tensor(destroy(Nc),idm)
b = tensor(idc,destroy(Nm))
num_a = num(Nc)
num_b = num(Nm)
comm_b = destroy(Nm)*create(Nm)-num_b
comm_a = destroy(Nc)*create(Nc)-num_a
#collapse operators
cc = sqrt(kappa)*a
cm = sqrt(gamma*(1.0 + n_th))*b
cp = sqrt(gamma*n_th)*b.dag()
c_ops = [cc,cm,cp]

print('Starting Computation...')
for row in range(dg_num):
     start_time = time.time()
     g0 = G0[row] 
     H1 = g0*(b.dag()+b)*(a.dag()*a) + b.dag()*b + E*(a.dag()+a)
     results = parfor(ss_iter,DW,num_cpus=5)
     cav_amp_data[row,:] = results[0]
     mech_amp_data[row,:] = results[1]
     pos_weight_data[row,:] = results[2]
     neg_weight_data[row,:] = results[3]
     fano_data[row,:] = results[4]
     mech_comm_data[row,:] = results[5]
     cav_comm_data[row,:] = results[6]
     cav_var_data[row,:] = results[7]
     print('Row ' + str(row) + ' done')
     print('Elapsed time: ',round(time.time()-start_time,1)) 

print('Computation Finished!')

