#
# Single photon source based on a three level atom strongly coupled to a cavity
#
# We follow the treatment presented in Kuhn et al., 
# Appl. Phys. B 69, 373-377 (1999),
# http://www.mpq.mpg.de/qdynamics/publications/library/APB69p373_Kuhn.pdf,
# for more details see M. Hennrich's thesis,
# http://mediatum2.ub.tum.de/node?id=602970.
#
# We study the following lambda system,
#
#                |e>
#             --------
#             /     \
#      Omega /       \ g
#           /         \
#          /        -------
#      -------        |g>
#       |u>
#
# where |u> and |g> are the ground states and |e> is the exicted state.
# |u> and |e> are coupled by a classical laser field with Rabi frequency
# Omega, and |g> and |e> by a cavity field with 2g being the single-photon
# Rabi frequency.
#
from __future__ import division

from qutip.Qobj import *
from qutip.tensor import *
from qutip.ptrace import *
from qutip.operators import *
from qutip.expect import *
from qutip.wigner import *
from qutip.mesolve import mesolve

from pylab import *

def run():

    # Define atomic states. Use ordering from paper
    ustate = basis(3,0)
    excited = basis(3,1)
    ground = basis(3,2)

    # Set where to truncate Fock state for cavity
    N = 2

    # Create the atomic operators needed for the Hamiltonian
    sigma_ge = tensor(qeye(N), ground * excited.dag()) # |g><e|
    sigma_ue = tensor(qeye(N), ustate * excited.dag()) # |u><e|

    # Create the photon operator
    a = tensor(destroy(N), qeye(3))
    ada = tensor(num(N), qeye(3))

    # Define collapse operators
    c_op_list = []
    # Cavity decay rate
    kappa = 1.5
    c_op_list.append(sqrt(kappa) * a)

    # Atomic decay rate
    gamma = 6
    # Use Rb branching ratio of 5/9 e->u, 4/9 e->g
    c_op_list.append(sqrt(5*gamma/9) * sigma_ue)
    c_op_list.append(sqrt(4*gamma/9) * sigma_ge)

    # Define time vector
    t = linspace(-15,15,100)
    # Define pump strength as a function of time
    wp = lambda t: 9 * exp(-(t/5)**2)

    # Set up the time varying Hamiltonian
    g = 5
    H0 = -g * (sigma_ge.dag() * a + a.dag() * sigma_ge)
    H1 = (sigma_ue.dag() + sigma_ue)
    def Hfunc(t, args):
        H0 = args[0]
        H1 = args[1]
        w = wp(t)
        return H0 - w * H1

    # Define initial state
    psi0 = tensor(basis(N,0), ustate)

    # Define states onto which to project (same as in paper)
    state_GG = tensor(basis(N,1), ground)
    sigma_GG = state_GG * state_GG.dag()
    state_UU = tensor(basis(N,0), ustate)
    sigma_UU = state_UU * state_UU.dag()

    output = mesolve(Hfunc, psi0, t, c_op_list,
                     [ada, sigma_UU, sigma_GG], [H0, H1])

    exp_ada, exp_uu, exp_gg = output.expect[0],output.expect[1],output.expect[2]

    # Plot the results
    fig=figure()
    subplot(211)
    plot(t, wp(t), 'k')
    ylabel('Control Field, $\Omega_\mathrm{p}$ [MHz]')
    ax = twinx()
    plot(t, kappa*exp_ada, 'b')
    ylabel('Cavity emission rate, $1/\mu s$')
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    subplot(212)
    plot(t, exp_uu, 'k-', label='$P{\mathrm{uu}}$')
    plot(t, exp_gg, 'k:',  label='$P{\mathrm{gg}}$')
    ylabel('Population')
    xlabel('Time [$\mu s$]')
    legend()
    show()
    
if __name__=='__main__':
    run()

