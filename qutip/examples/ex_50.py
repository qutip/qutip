#
# Nonadiabatic sweep: Gradually transform a simple decoupled spin chain 
# hamiltonian to a complicated interacting spin chain.
#

from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.states import *
from qutip.odesolve import mesolve

from pylab import *

import time

def compute(N, M, h, Jx, Jy, Jz, taulist):

    # pre-allocate operators
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))
        
    #
    # Construct the initial hamiltonian and state vector
    #
    psi_list = [basis(2,0) for n in range(N)]
    psi0 = tensor(psi_list)
    H0 = 0    
    for n in range(N):
        H0 += - 0.5 * 2.5 * sz_list[n]

    #
    # Construct the target hamiltonian
    #

    # energy splitting terms
    H1 = 0    
    for n in range(N):
        H1 += - 0.5 * h[n] * sz_list[n]

    H1 = 0    
    for n in range(N-1):
        # interaction terms
        H1 += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
        H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
        H1 += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]


    # the time-dependent hamiltonian in list-string format
    h_t = [[H0, '(%f-t)/%f' % (max(taulist),max(taulist))], [H1, 't/%f' % max(taulist)]]       

    #
    # callback function for each time-step
    #
    evals_mat      = zeros((len(taulist),M))
    occupation_mat = zeros((len(taulist),M))

    idx = [0]
    def process_rho(tau, psi):
  
        # evaluate the Hamiltonian with gradually switched on interaction
        H = qobj_list_evaluate(h_t, tau, {})

        # find the M lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=M)

        evals_mat[idx[0],:] = real(evals)
    
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            occupation_mat[idx[0],n] = abs(dot(eket.dag().data, psi.data)[0,0])**2    
        
        idx[0] += 1 
        
    #
    # Evolve the system, request the solver to call process_rho at each time
    # step.
    #
    mesolve(h_t, psi0, taulist, [], process_rho)
        
    return evals_mat, occupation_mat
    
def run():

    #
    # set up the paramters
    #
    N = 6            # number of spins
    M = 20           # number of eigenenergies to solve for

    # array of spin energy splittings and coupling strengths (random values). 
    h  = 1.0 * 2 * pi * (1 - 2 * rand(N))
    Jz = 1.0 * 2 * pi * (1 - 2 * rand(N))
    Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))
    Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))

    # increase taumax to get make the sweep more adiabatic
    taumax = 5.0
    taulist = linspace(0, taumax, 100)

    start_time = time.time()
    evals_mat, occ_mat = compute(N, M, h, Jx, Jy, Jz, taulist)
    print 'time elapsed = ' +str(time.time() - start_time) 

    #---------------------------------------------------------------------------
    # plots
    #
    rc('text', usetex=True)
    rc('font', family='serif')

    figure(figsize=(9,12))

    #
    # plot the energy eigenvalues
    #
    subplot(2,1,1)
    
    # first draw thin lines outlining the energy spectrum
    for n in range(len(evals_mat[0,:])):
        if n == 0:
            ls = 'b'
            lw = 1        
        else:
            ls = 'k'        
            lw = 0.25        
        plot(taulist/max(taulist), evals_mat[:,n] / (2*pi), ls, linewidth=lw)

    # second, draw line that encode the occupation probability of each corresponding
    # state in the linewidth. thicker line => high occupation probability.
    for idx in range(len(taulist)-1):
        for n in range(len(occ_mat[0,:])):
            lw = 0.5 + 4*occ_mat[idx,n]    
            if lw > 0.55:
                plot(array([taulist[idx], taulist[idx+1]])/taumax, array([evals_mat[idx,n], evals_mat[idx+1,n]])/(2*pi), 'r', linewidth=lw)    
        
    xlabel(r'$\tau$')
    ylabel('Eigenenergies')
    title("Energyspectrum (%d lowest values) of a chain of %d spins.\nThe occupation probabilities are encoded in the red line widths." % (M, N))
    legend(("Ground state",))

    #
    # plot the occupation probabilities for the few lowest eigenstates
    #
    subplot(2,1,2)
    for n in range(len(occ_mat[0,:])):
        if n == 0:
            plot(taulist/max(taulist), 0 + occ_mat[:,n], 'r', linewidth=2)
        else:
            plot(taulist/max(taulist), 0 + occ_mat[:,n])

    xlabel(r'$\tau$')
    ylabel('Occupation probability')
    title("Occupation probability of the %d lowest eigenstates for a chain of %d spins" % (M, N))
    legend(("Ground state",))

    show()

if __name__=='__main__':
    run()

