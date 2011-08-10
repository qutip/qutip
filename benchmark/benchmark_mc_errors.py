#
# MC solver error estimates: solve a system with increasingly number of MC 
# trajectories and record the errors (compared to the "exact" results from the
# ODE solver). 
#

# disable the MC progress bar
import os
os.environ['QUTIP_GRAPHICS']="NO"

from qutip import *
from pylab import *
import time

def system_integrate(Na, Nb, wa, wb, wab, ga, gb, solver, ntraj):

    # Hamiltonian and initial state
    a = tensor(destroy(Na), qeye(Nb))
    b = tensor(qeye(Na), destroy(Nb))
    na = a.dag() * a
    nb = b.dag() * b
    H = wa * na  + wb * nb + wab * (a.dag() * b + a * b.dag())

    # start with one more excitation in a than in b
    psi0 = tensor(basis(Na,Na-1), basis(Nb,Nb-2))

    # collapse operators
    c_op_list = []

    rate = ga
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = gb
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * b)

    if solver == "me":
        expt_list = odesolve(H, psi0 * psi0.dag(), tlist, c_op_list, [na, nb])
    elif solver == "wf":
        expt_list = odesolve(H, psi0, tlist, [], [na, nb])
    elif solver == "es":
        expt_list = essolve(H, psi0, tlist, c_op_list, [na, nb])
    elif solver == "mc":
        expt_list = mcsolve(H, psi0, tlist, ntraj, c_op_list, [na, nb])
    else:
        raise ValueError("unknown solver")
        
    return expt_list[0], expt_list[1]
    
#
# set up the calculation
#
wa  = 1.0 * 2 * pi   # frequency of system a
wb  = 1.0 * 2 * pi   # frequency of system a
wab = 0.1 * 2 * pi   # coupling frequency
ga = 0.50            # dissipation rate of system a
gb = 0.25            # dissipation rate of system b
Na = 2               # number of states in system a
Nb = 2               # number of states in system b

tlist = linspace(0, 10, 100)

ntraj_vec = array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500])

na_sse = zeros(len(ntraj_vec))
nb_sse = zeros(len(ntraj_vec))

na_se = zeros((len(ntraj_vec), len(tlist)))
nb_se = zeros((len(ntraj_vec), len(tlist)))

n_runs = 10

for n_run in range(n_runs):
    print "run number %d" % n_run
    n_idx = 0
    for ntraj in ntraj_vec:   

        start_time = time.time()
        na_ode, nb_ode = system_integrate(Na, Nb, wa, wb, wab, ga, gb, "me", 0)
        print "ME solver elapsed time: ", time.time() - start_time

        start_time = time.time()
        na_mc, nb_mc = system_integrate(Na, Nb, wa, wb, wab, ga, gb, "mc", ntraj)
        print "MC[%d] solver elapsed time: %s" % (ntraj, str(time.time() - start_time))

        #figure(1)
        #plot(tlist, real(na_ode), 'r', tlist, real(nb_ode), 'r')    
        #plot(tlist, real(na_mc),  'r.', tlist, real(nb_mc), 'b.')    
        #show()

        na_sse[n_idx] += sum(abs(na_ode-na_mc)) / len(tlist)
        nb_sse[n_idx] += sum(abs(nb_ode-nb_mc)) / len(tlist)

        na_se[n_idx, :] += abs(na_ode-na_mc)
        nb_se[n_idx, :] += abs(nb_ode-nb_mc)

        n_idx += 1

#
# calculate the averages and save the data
#
na_sse = na_sse / n_runs 
nb_sse = nb_sse / n_runs

na_se = na_se / n_runs 
nb_se = nb_se / n_runs

sse_mat = array([ntraj_vec, na_sse, nb_sse]).T
file_data_store("benchmark-mc-errors-vs-ntraj-Na-%d-Nb-%d-runs-%d.dat" % (Na, Nb, n_runs), sse_mat, "real")

na_se_mat = append(matrix(tlist).T, na_se.T, axis=1)
file_data_store("benchmark-mc-na-errors-time-Na-%d-Nb-%d-runs-%d.dat" % (Na, Nb, n_runs), na_se_mat, "real")

nb_se_mat = append(matrix(tlist).T, nb_se.T, axis=1)
file_data_store("benchmark-mc-na-errors-time-Na-%d-Nb-%d-runs-%d.dat" % (Na, Nb, n_runs), nb_se_mat, "real")



#
# plot benchmark data
#
figure(2)
plot(ntraj_vec, na_sse, 'r')  
plot(ntraj_vec, nb_sse, 'b')  

xlabel('ntraj')
ylabel('SSE')
title('MC errors vs ntraj')
savefig("benchmark-mc-errors-vs-ntraj.pdf")


#
# plot errors as a function of time
#

figure(3)
for idx in range(len(ntraj_vec)):
    plot(tlist, na_se[idx, :], 'r')  
    plot(tlist, nb_se[idx, :], 'b')  

xlabel('time')
ylabel('average squared errors')
title('MC errors vs time')
#savefig("benchmark-mc-errors-vs-ntraj.pdf")



show()



