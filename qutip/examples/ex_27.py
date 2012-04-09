#
# Qubit dynamics shown in a Bloch sphere.
#
from qutip.tensor import *
from qutip.operators import *
from qutip.expect import *
from qutip.states import basis
from qutip.odesolve import mesolve
from qutip.Bloch import *

from pylab import *
import time


def qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist):
    # Hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * (cos(theta) * sz + sin(theta) * sx)
    # collapse operators
    c_op_list = []
    n_th = 0 # zero temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)
    # evolve and calculate expectation values
    expt_list = mesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])  
    return expt_list[0], expt_list[1], expt_list[2]

def run():   

    w     = 1.0 * 2 * pi  # qubit angular frequency
    theta = 0.2 * pi      # qubit angle from sigma_z axis (toward sigma_x axis)
    gamma1 = 0.05         # qubit relaxation rate
    gamma2 = 1.0          # qubit dephasing rate

    # initial state
    a = .5
    psi0 = (a* basis(2,0) + (1-a)*basis(2,1)).unit()

    tlist = linspace(0,3,500)
    sx, sy, sz = qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist)

    sphere=Bloch()
    sphere.add_points([sx,sy,sz])
    sphere.point_color=['r']
    sphere.vector_color = ['b']
    sphere.size=[4,4]
    sphere.font_size=14
    sphere.add_vectors([sin(theta),0,cos(theta)])
    sphere.show()
    

if __name__=="__main__":
    run()

