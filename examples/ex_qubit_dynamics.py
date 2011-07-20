#
# Textbook example: Rabi oscillation in the dissipative Jaynes-Cummings model.
# 
#
from qutip import *
from pylab import *
import time

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

    if solver == "ode":
        expt_list = odesolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    elif solver == "es":
        expt_list = essolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    elif solver == "mc":
        ntraj = 250
        expt_list = mcsolve(H, psi0, tlist, ntraj, c_op_list, [sigmax(), sigmay(), sigmaz()])  
    else:
        raise ValueError("unknown solver")
        
    return expt_list[0], expt_list[1], expt_list[2]
    
#
# set up the calculation
#
epsilon = 0.0 * 2 * pi   # cavity frequency
delta   = 1.0 * 2 * pi   # atom frequency
g2 = 0.05
g1 = 0.000

# intial state
psi0 = basis(2,0)

tlist = linspace(0,5,200)

start_time = time.time()
sx, sy, sz = qubit_integrate(epsilon, delta, g1, g2, "ode")
print 'time elapsed = ' +str(time.time() - start_time) 

plot(tlist, sx, 'r')
plot(tlist, sy, 'b')
plot(tlist, sz, 'g')

sx_analytic = zeros(shape(tlist))
sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

plot(tlist, sx_analytic, 'r*')
plot(tlist, sy_analytic, 'g*')
plot(tlist, sz_analytic, 'g*')

print "sx error =", max(abs(sx - sx_analytic))
print "sy error =", max(abs(sy - sy_analytic))
print "sy error =", max(abs(sz - sz_analytic))


legend(("sx", "sy", "sz"))
xlabel('Time')
ylabel('expectation value')
show()


