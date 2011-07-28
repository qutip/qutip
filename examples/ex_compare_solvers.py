#
# Textbook example: Rabi oscillation in the dissipative Jaynes-Cummings model.
# 
#
from qutip import *
from pylab import *
import time

import warnings
warnings.simplefilter("error", np.ComplexWarning)


def qubit_integrate(epsilon, delta, g1, g2, solver):

    H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()
        
    # collapse operators
    c_op_list = []

    rate = g1
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sigmam())

    rate = g2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sigmaz())

    if solver == "me":
        expt_list = odesolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    elif solver == "wf":
        expt_list = odesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])  
    elif solver == "es":
        expt_list = essolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    elif solver == "mc1":
        expt_list = mcsolve(H, psi0, tlist, 1, [], [sigmax(), sigmay(), sigmaz()])  
    elif solver == "mc250":
        expt_list = mcsolve(H, psi0, tlist, 250, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    else:
        raise ValueError("unknown solver")
        
    return expt_list[0], expt_list[1], expt_list[2]
    
#
# set up the calculation
#
epsilon = 0.0 * 2 * pi   # cavity frequency
delta   = 1.0 * 2 * pi   # atom frequency
g2 = 0.0
g1 = 0.1

# intial state
psi0 = basis(2,0)

tlist = linspace(0,5,200)

for solver in ("me", "es", "mc1", "mc250"):

    start_time = time.time()
    sx1, sy1, sz1 = qubit_integrate(epsilon, delta, g1, g2, solver)
    print solver + ' time elapsed = ' +str(time.time() - start_time) 

    figure(1)
    plot(tlist, real(sx1), 'r')
    plot(tlist, real(sy1), 'b')
    plot(tlist, real(sz1), 'g')
    
    #legend_vec += ("sx


#legend(("sx", "sy", "sz"))
xlabel('Time')
ylabel('Expectation value')

show()




