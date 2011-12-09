#
# Textbook example: Landau-Zener transitions in a quantum two-level system.
#
from qutip import *
from pylab import *
import time

def qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz
        
    H = [[H0, H1],['w*t']]

    # collapse operators
    c_op_list = []

    n_th = 0.0 # zero temperature

    # relaxation
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # excitation
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())

    # dephasing 
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)
    rhs_generate(H,H_args={'w':1.0})
    # evolve and calculate expectation values
    opts=Odeoptions()
    opts.rhs_reuse=True
    data=[]
    for w in linspace(0,1,10):
        data.append(odesolve(H, psi0,tlist,c_op_list,[sm.dag() * sm],H_args={'w':w},options=opts))
    return data[-2][0]
    
#
# set up the calculation
#
delta = 0.5 * 2 * pi   # qubit sigma_x coefficient
eps0  = 0.1 * 2 * pi   # qubit sigma_z coefficient
A     = 2.0 * 2 * pi   # sweep rate
gamma1 = 0.0           # relaxation rate
gamma2 = 0.0        # dephasing  rate
psi0 = basis(2,0)      # initial state

tlist = linspace(-10.0, 10.0, 1500)

start_time = time.time()
p_ex = qubit_integrate(delta, eps0, A, gamma1, gamma2, psi0, tlist)
print 'time elapsed = ' + str(time.time() - start_time) 
plot(tlist, p_ex, 'b', tlist, 1-p_ex, 'r')
plot(tlist, 1 - exp( - pi * delta **2 / (2 * A)) * ones(shape(tlist)), 'k')
xlabel('Time')
ylabel('Occupation probability')
title('Landau-Zener transition')
legend(("Excited state", "Ground state", "Landau-Zener formula"), loc=6)
show()


