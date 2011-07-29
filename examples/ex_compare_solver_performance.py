#
# Benchmark for solvers.
# 
#
from qutip import *
from pylab import *
import time

import warnings
warnings.simplefilter("error", np.ComplexWarning)


def system_integrate(Na, Nb, wa, wb, wab, ga, gb, solver):

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
    elif solver == "mc1":
        expt_list = mcsolve(H, psi0, tlist, 1, c_op_list, [na, nb])
    elif solver == "mc250":
        expt_list = mcsolve(H, psi0, tlist, 250, c_op_list, [na, nb])
    elif solver == "mc500":
        expt_list = mcsolve(H, psi0, tlist, 500, c_op_list, [na, nb])
    else:
        raise ValueError("unknown solver")
        
    return expt_list[0], expt_list[1]
    
#
# set up the calculation
#
wa  = 1.0 * 2 * pi   # frequency of system a
wb  = 1.0 * 2 * pi   # frequency of system a
wab = 0.1 * 2 * pi   # coupling frequency
ga = 0.0             # dissipation rate of system a
gb = 0.0             # dissipation rate of system b
Na = 2               # number of states in system a
Nb = 2               # number of states in system b

tlist = linspace(0, 10, 200)

show_dynamics = False
style_map = {"es": "r.", "ode": "b", "mc1": "g", "wf": "m*"}
#
# First test unitary solvers
#
solvers = ("wf", "es", "mc1")
Na_vec = arange(2, 60, 1)

times = zeros((len(Na_vec), len(solvers)))

n_runs = 1

for n_run in range(n_runs):
    print "run number %d" % n_run
    n_idx = 0
    for Na in Na_vec:   
        print "using %d states" % (Na * Nb)
        s_idx = 0
        for solver in solvers:
            print "solver " + solver
            start_time = time.time()
            na, nb = system_integrate(Na, Nb, wa, wb, wab, ga, gb, solver)
            times[n_idx, s_idx] += time.time() - start_time
            s_idx += 1

            if show_dynamics:
                figure(3)
                plot(tlist, real(na), style_map[solver], tlist, real(nb), style_map[solver])    

        if show_dynamics:
            show()

        n_idx += 1

times = times / n_runs

#
# plot benchmark data
#
print "times =", times

figure(1)
s_idx = 0

for solver in solvers:
        
    plot(Na_vec * Nb, times[:,s_idx])  
    s_idx += 1

xlabel('Numbers of quantum states')
ylabel('Time to evolve system (seconds)')
title('Comparison of solver performance for unitary evolution')
legend(solvers)
savefig("solver-performance-unitary.png")

show()


#
# Second, test nonunitary solvers solvers
#
ga = 0.05            # dissipation rate of system a
gb = 0.0             # dissipation rate of system b

solvers = ("me", "mc250", "mc500", "es")
Na_vec = arange(2, 35, 1)

show_dynamics = True

times = zeros((len(Na_vec), len(solvers)))

n_runs = 1

for n_run in range(n_runs):
    print "run number %d" % n_run
    n_idx = 0
    for Na in Na_vec:   
        print "using %d states" % (Na * Nb)
        s_idx = 0
        for solver in solvers:
        
            start_time = time.time()
            na, nb = system_integrate(Na, Nb, wa, wb, wab, ga, gb, solver)
            times[n_idx, s_idx] += time.time() - start_time
            s_idx += 1

            if show_dynamics:
                figure(3)
                plot(tlist, real(na), 'r', tlist, real(nb), 'b')    

        if show_dynamics:
            show()

        n_idx += 1

times = times / n_runs


#
# plot benchmark data
#
print "times =", times

figure(2)
s_idx = 0

for solver in solvers:
        
    plot(Na_vec * Nb, times[:,s_idx])  
    s_idx += 1

xlabel('Numbers of quantum states')
ylabel('Time to evolve system (seconds)')
title('Comparison of solver performance for nonunitary evolution')
legend(solvers)
savefig("solver-performance-nonunitary.png")

