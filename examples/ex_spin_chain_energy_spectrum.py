#
# Energy spectrum for a Heisenberg spin 1/2 chain with interaction gradually
# switched on
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

    # construct the hamiltonian

    # energy splitting terms
    H0 = 0    
    for n in range(N):
        H0 += - 0.5 * h[n] * sz_list[n]

    # interaction terms
    H1 = 0    
    for n in range(N-1):
        H1 += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
        H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
        H1 += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]

    evals_mat = zeros((len(taulist),M))
    for idx, tau in enumerate(taulist):

        # evaluate the Hamiltonian with gradually switched on interaction
        #H = (1-tau) * H0 + tau * H1
        H = H0 + tau * H1

        # find the M lowest eigenvalues of the system
        evals = H.eigenenergies(eigvals=M)

        evals_mat[idx,:] = real(evals)

        idx += 1

    return evals_mat
    
#
# set up the calculation
#
N = 8            # number of spins
M = 20           # number of eigenenergies to solve for

# array of spin energy splittings and coupling strengths. here we use
# random parameters
h  = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jz = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))

taulist = linspace(0, 1, 100)

start_time = time.time()
evals_mat = run(N, M, h, Jx, Jy, Jz, taulist)
print 'time elapsed = ' +str(time.time() - start_time) 

#
# plot
#
rc('text', usetex=True)
rc('font', family='serif')

#
# plot the energy eigenvalues
#
figure(1)
for n in range(len(evals_mat[0,:])):
    if n == 0:
        ls = 'r'
        lw = 2
    else:
        ls = 'b'
        lw = 1

    #plot(taulist, (evals_mat[:,n]-evals_mat[:,0]) / (2*pi), ls, linewidth=lw)
    plot(taulist, evals_mat[:,n] / (2*pi), ls, linewidth=lw)

xlabel(r'$\tau$')
ylabel('Eigenenergies')
title("Energyspectrum (%d lowest values) of a chain of %d spins" % (M, N))

show()

