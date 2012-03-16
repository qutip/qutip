#
# Adiabatic sweep: find the ground state occupation probablility and the lowest
# energy spectrum
#
from qutip import *
from pylab import *
import time

def run(N, M, h, Jx, Jy, Jz, taulist):

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

    # interaction terms
    H1 = 0    
    for n in range(N-1):
        H1 += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
        H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
        H1 += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]

    #
    # Evolve the system
    #
    h_t = [[H0, '(%f-t)/%f' % (max(taulist),max(taulist))], [H1, 't/%f' % max(taulist)]]       
    psi_list = mesolve(h_t, psi0, taulist, [], [])

    evals_mat      = zeros((len(taulist),M))
    occupation_mat = zeros((len(taulist),M))
    for idx, tau in enumerate(taulist):

        # evaluate the Hamiltonian with gradually switched on interaction
        # XXX: use h_t to evaluate H ?
        H = (max(taulist) - tau)/max(taulist) * H0 + tau/max(taulist) * H1

        # find the M lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=M)

        evals_mat[idx,:] = real(evals)
    
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            #occupation_mat[idx,n] = (eket.dag() *  psi_list[idx]).full()[0,0]
            occupation_mat[idx,n] = expect(ket2dm(eket), psi_list[idx])
        
    return evals_mat, occupation_mat
    
#
# set up the calculation
#
N = 5            # number of spins
M = 20           # number of eigenenergies to solve for

# array of spin energy splittings and coupling strengths (random values). 
h  = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jz = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))

# increase taumax to get make the sweep more adiabatic
taumax = 10.0
taulist = linspace(0, taumax, 100)

start_time = time.time()
evals_mat, occ_mat = run(N, M, h, Jx, Jy, Jz, taulist)
print 'time elapsed = ' +str(time.time() - start_time) 

#-------------------------------------------------------------------------------
# plots
#
rc('text', usetex=True)
rc('font', family='serif')

figure(figsize=(9,12))

#
# plot the energy eigenvalues
#
subplot(2,1,1)
for n in range(len(evals_mat[0,:])):
    if n == 0:
        ls = 'r'
        lw = 2
    else:
        ls = 'b'
        lw = 1

    plot(taulist/max(taulist), evals_mat[:,n] / (2*pi), ls, linewidth=lw)

xlabel(r'$\tau$')
ylabel('Eigenenergies')
title("Energyspectrum (%d lowest values) of a chain of %d spins" % (M, N))

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

