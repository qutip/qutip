#
# Using the propagator to find the steady state of a driven system.
#
from qutip.states import *
from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.propagator import *

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def run():

    #
    # Define a hamiltonian: a strongly driven two-level system
    # (repeated LZ transitions)
    #
    def hamiltonian_t_bk(t, args):
        """ evaluate the hamiltonian at time t"""
        H0 = args[0]
        H1 = args[1]
        w  = args[2]
    
        return H0 + cos(w * t) * H1

    #
    # configure the parameters 
    #
    delta = 0.075 * 2 * pi  # qubit sigma_x coefficient
    eps0  = 0.0   * 2 * pi  # qubit sigma_z coefficient
    A     = 2.0   * 2 * pi  # sweep rate
    gamma1 = 0.0001        # relaxation rate
    gamma2 = 0.005         # dephasing  rate
    psi0   = basis(2,0)    # initial state
    omega  = 0.05 * 2 * pi # driving frequency
    T      = (2*pi)/omega  # driving period

    #
    # Hamiltonian
    #
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A/2.0 * sz
    args = {'w': omega}
            
    def H1_coeff_t(t, args):
        return cos(args['w'] * t)
           
    hamiltonian_t = [H0, [H1, H1_coeff_t]]
    
    # hamiltonian_t = [H0, [H1, 'cos(w * t)']] # using the list-string format

    #
    # collapse operators
    #
    c_op_list = []

    n_th = 0.0 # zero temperature

    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)       # relaxation

    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag()) # excitation

    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)       # dephasing 


    #
    # evolve for five driving periods
    #
    tlist = linspace(0.0, 5 * T, 1500)
    p_ex = mesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], args)  

    #
    # find the propagator for one driving period
    #
    T = 2*pi / omega
    U = propagator(hamiltonian_t, T, c_op_list, args)

    #
    # find the steady state of repeated applications of the propagator
    # (i.e., t -> inf)
    #
    rho_ss  = propagator_steadystate(U)
    p_ex_ss = real(expect(sm.dag() * sm, rho_ss))

    #
    # plot the results
    #
    figure(1)

    subplot(211)
    plot(tlist, real(p_ex[0]), 'b')
    plot(tlist, real(1-p_ex[0]), 'r')
    plot(tlist, ones(shape(tlist)) * p_ex_ss, 'k', linewidth=2)
    xlabel('Time')
    ylabel('Probability')
    title('Occupation probabilty of qubit [NEW]')
    legend((r"$\left|1\right>$", r"$\left|0\right>$", r"$\left|1\right>$ steady state"), loc=0)

    subplot(212)
    plot(tlist, -delta/2.0 * ones(shape(tlist)), 'r')
    plot(tlist, -(eps0/2.0 + A/2.0 * cos(omega * tlist)), 'b')
    legend(("$\sigma_x$ coefficient", "$\sigma_z$ coefficient"))
    xlabel('Time')
    ylabel('Coefficients in the Hamiltonian')

    show()


if __name__=='__main__':
    run()

