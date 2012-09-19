#
# Landau-Zener-Stuckelberg interferometry: steady state of repeated Landau-Zener
# like avoided-level crossing, as a function of driving amplitude and bias.
#
# Note: In order to get this example to work properly in the demos window,
# we have had to pass many more variables to parfor than is typically necessary.

from qutip import *
from pylab import *

# a task function for the for-loop parallelization: 
# the m-index is parallelized in loop over the elements of p_mat[m,n]
def task(args):
    m, H_td, c_op_list, sn, A_list, T, w, eps = args
    p_mat_m = zeros(len(A_list))
    for n, A in enumerate(A_list):
        # change args sent to solver, w is really a constant though.
        Hargs = {'w': w, 'eps': eps,'A': A}
        # settings (for reusing list-str format Hamiltonian)
        U = propagator(H_td, T, c_op_list, Hargs, Odeoptions(rhs_reuse = True))
        rho_ss = propagator_steadystate(U)
        p_mat_m[n] = expect(sn, rho_ss)
    return [m, p_mat_m]

def run():
    
    # set up the parameters and start calculation
    delta    = 0.1  * 2 * pi  # qubit sigma_x coefficient
    w        = 2.0  * 2 * pi  # driving frequency
    T        = 2 * pi / w     # driving period 
    gamma1   = 0.00001        # relaxation rate
    gamma2   = 0.005          # dephasing  rate
    eps_list = linspace(-10.0, 10.0, 101) * 2 * pi
    A_list   = linspace(0.0, 20.0, 101) * 2 * pi

    # pre-calculate the necessary operators
    sx = sigmax(); sz = sigmaz(); sm = destroy(2); sn = num(2)
    # collapse operators
    c_op_list = [sqrt(gamma1) * sm, sqrt(gamma2) * sz]#relaxation and dephasing

    # setup time-dependent Hamiltonian (list-string format)
    H0 = -delta/2.0 * sx
    H1 = [sz,'-eps/2.0+A/2.0*sin(w * t)']
    H_td = [H0,H1]
    Hargs = {'w': w,'eps':eps_list[0],'A':A_list[0]}

    # pre-generate RHS so we can use parfor
    rhs_generate(H_td,c_op_list,Hargs,name='lz_func')
    # start a parallel for loop over bias point values (eps_list)
    parfor_args=[[k,H_td,c_op_list,sn,A_list,T,w,eps_list[k]] for k in range(len(eps_list))]
    p_mat_list = parfor(task, parfor_args)

    # assemble a matrix p_mat from list of (index,array) tuples returned by parfor
    p_mat = zeros((len(eps_list),len(A_list)))
    for m, p_mat_m in p_mat_list:
        p_mat[m,:] = p_mat_m

    # Plot the results
    A_mat, eps_mat = meshgrid(A_list/(2*pi), eps_list/(2*pi))
    fig = figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.8])
    c = ax.pcolor(eps_mat,A_mat,p_mat)
    c.set_cmap('RdYlBu_r')
    cbar = fig.colorbar(c)
    cbar.set_label("Probability")
    ax.set_xlabel(r'Bias point $\epsilon$')
    ax.set_ylabel(r'Amplitude $A$')
    ax.autoscale(tight=True)
    title('Steadystate excitation probability\n' + 
          '$H = -\\frac{1}{2}\\Delta\\sigma_x -\\frac{1}{2}\\epsilon\\sigma_z'+
          ' - \\frac{1}{2}A\\sin(\\omega t)$\n')
    show()

if __name__=='__main__':
    run()
