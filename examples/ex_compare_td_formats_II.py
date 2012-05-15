#
# Test cases for time-dependent collapse operators
#
from qutip import *
from pylab import *
import time

def gamma_t(t, args):
    """ a time-dependent rate """
    return args['G'] * (1 + sin(args['w']*t)) * exp(-0.5*t)

def H1_coeff_t(t, args):
    return sin(args['w'] * t)

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]
    w  = args[2]
    return H0 + H1 * sin(w * t)

def qubit_integrate(solver, N, w0, A, w, gamma, rho0, tlist):

    # Hamiltonian: oscillator + driving on x
    H0 = w0 * 2 * pi * num(N)
    H1 = A * (destroy(N).dag() + destroy(N))

    # --------------------------------------------------------------------------
    # 1) time-dependent hamiltonian and collapse operators: using list-function
    #    format
    #
    
    H = [H0, [H1, H1_coeff_t]]
    args = {'w': w, 'G': gamma}

    c_op_list = []
    c_op_list.append([destroy(N), gamma_t]) # relaxation
    
    start_time = time.time()
    output = solver(H, rho0, tlist, c_op_list, [num(N)], args=args)  
    expt_list1 = output.expect
    print 'Method 1: time elapsed = ' + str(time.time() - start_time) 
        
    # --------------------------------------------------------------------------
    # 2) time-dependent hamiltonian and collapse operators: using list-string
    #    format
    #
    
    H = [H0, [H1, 'sin(w * t)']]
    args = {'w': w, 'G': gamma}

    c_op_list = []
    c_op_list.append([destroy(N), 'G * (1 + sin(w*t))']) # relaxation

    start_time = time.time()
    output = solver(H, rho0, tlist, c_op_list, [num(N)], args=args)
    expt_list2 = output.expect      
    print 'Method 2: time elapsed = ' + str(time.time() - start_time)  
    
    # --------------------------------------------------------------------------
    # 3) time-dependent hamiltonian and but time-independent collapse operators:
    #    using function callback format
    #
    args = [H0, H1, w]

    c_op_list = []
    c_op_list.append(sqrt(gamma) * destroy(N)) # relaxation

    start_time = time.time()
    output = solver(hamiltonian_t, rho0, tlist, c_op_list, [num(N)], args=args)      
    expt_list3 = output.expect
    print 'Method 3: time elapsed = ' + str(time.time() - start_time)         

    # --------------------------------------------------------------------------
    # 4) Constant hamiltonian and collapse operators
    #
    c_op_list = []
    c_op_list.append(sqrt(gamma) * destroy(N)) # relaxation

    start_time = time.time()
    output = solver(H0, rho0, tlist, c_op_list, [num(N)])  
    expt_list4 = output.expect
    print 'Method 4: time elapsed = ' + str(time.time() - start_time)     
    
    # --------------------------------------------------------------------------
    # 5) Unitary evolution, constant hamiltonian
    #
    start_time = time.time()
    output = mesolve(H0, rho0, tlist*10, [], [num(N)])  
    expt_list5 = output.expect
    print 'Method 5: time elapsed = ' + str(time.time() - start_time)         

    return expt_list1[0], expt_list2[0], expt_list3[0], expt_list4[0], expt_list5[0]
    
#
# parameters
#
N     = 10
w0    = 1.0 * 2 * pi   # oscillator energy
A     = 0.15 * 2 * pi  # driving amplitude (reduce to make the RWA more accurate)
w     = 1.0 * 2 * pi   # driving frequency
gamma = 0.2            # relaxation rate
rho0 = fock_dm(N,N-2)  # initial state

tlist = linspace(0, 10.0, 250)

for solver in [mesolve]: #[mesolve, mcsolve]:
    p_ex1, p_ex2, p_ex3, p_ex4, p_ex5 = qubit_integrate(solver, N, w0, A, w, gamma, rho0, tlist)

    figure()
    plot(tlist, real(p_ex1), 'b', tlist, real(p_ex2), 'g.', tlist, real(p_ex3), 'r.-', tlist, real(p_ex4), 'c--', tlist, real(p_ex5), 'm')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    legend(("Time-dependent function format", "Time-dependent string format", "Time-dependent func format. Const collapse ops.", "Const. collapse operators", "Unitary evolution"))

show()


