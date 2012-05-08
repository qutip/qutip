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
    output = mesolve(h_t, psi0, taulist, [], [])

    evals_mat      = zeros((len(taulist),M))
    occupation_mat = zeros((len(taulist),M))
    for idx, tau in enumerate(taulist):

        # evaluate the Hamiltonian with gradually switched on interaction
        # XXX: use h_t to evaluate H ?
        H = (max(taulist) - tau)/max(taulist) * H0 + tau/max(taulist) * H1
        #H = qobj_list_evaluate(h_t, tau, {})

        # find the M lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=M)

        evals_mat[idx,:] = real(evals)
    
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            #occupation_mat[idx,n] = abs((eket.dag() *  psi_list[idx]).full()[0,0])**2
            occupation_mat[idx,n] = abs(dot(eket.dag().data, output.states[idx].data)[0,0])**2
        
    return evals_mat, occupation_mat
    
#
# set up the calculation
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


#
# fancier way of plotting the same thing... also illustrates which eigenstates
# are occupied as a function of time
#
figure(2)

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
show()

