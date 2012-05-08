#
# Example comparing master equation with constant and time-dependent collapse
# Hamiltonian and collapse operators.
#
from qutip import *
from pylab import *
import time

def gamma1_t(t, args):
    """ a time-dependent rate """
    return args['G1'] * exp(-0.1*t)
    
def gamma2_t(t, args):
    """ a time-dependent rate """
    return args['G2'] * exp(-0.2*t)

def H1_coeff_t(t, args):
    return sin(args['w'] * t)

def hamiltonian_t(t, args):
    """ evaluate the hamiltonian at time t. """
    H0 = args[0]
    H1 = args[1]
    w  = args[2]
    return H0 + H1 * sin(w * t)

def qubit_integrate(delta, eps0, A, w, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = - delta/2.0 * sx - eps0/2.0 * sz
    H1 = - A * sx

    # --------------------------------------------------------------------------
    # 1) time-dependent hamiltonian and collapse operators: using list-function
    #    format
    #
    
    H = [H0, [H1, H1_coeff_t]]
    args = {'w': w, 'G1': gamma1, 'G2': gamma2}

    c_op_list = []
    c_op_list.append([sm, gamma1_t]) # relaxation
    c_op_list.append([sz, gamma2_t]) # dephasing
    #c_op_list.append(sqrt(gamma1) * sm) # relaxation
    #c_op_list.append(sqrt(gamma2) * sz) # dephasing
    
    start_time = time.time()
    output = mesolve(H, psi0, tlist, c_op_list, [sm.dag() * sm], args=args)  
    expt_list1 = output.expect
    print 'Method 1: time elapsed = ' + str(time.time() - start_time) 
        
    # --------------------------------------------------------------------------
    # 2) time-dependent hamiltonian and collapse operators: using list-string
    #    format
    #
    
    H = [H0, [H1, 'sin(w * t)']]
    args = {'w': w, 'G1': gamma1, 'G2': gamma2}

    c_op_list = []
    c_op_list.append([sm, 'G1 * exp(-0.1*t)']) # relaxation
    c_op_list.append([sz, 'G2 * exp(-0.2*t)']) # dephasing
    #c_op_list.append(sqrt(gamma1) * sm) # relaxation
    #c_op_list.append(sqrt(gamma2) * sz) # dephasing

    start_time = time.time()
    output = mesolve(H, psi0, tlist, c_op_list, [sm.dag() * sm], args=args)
    expt_list2 = output.expect      
    print 'Method 2: time elapsed = ' + str(time.time() - start_time)  
    
    # --------------------------------------------------------------------------
    # 3) time-dependent hamiltonian and but time-independent collapse operators:
    #    using function callback format
    #
    
    args = [H0, H1, w]

    c_op_list = []
    c_op_list.append(sqrt(gamma1) * sm) # relaxation
    c_op_list.append(sqrt(gamma2) * sz) # dephasing

    start_time = time.time()
    output = mesolve(hamiltonian_t, psi0, tlist, c_op_list, [sm.dag() * sm], args=args)      
    expt_list3 = output.expect
    print 'Method 3: time elapsed = ' + str(time.time() - start_time)         

    # --------------------------------------------------------------------------
    # 4) Constant hamiltonian and collapse operators
    #

    H_rwa = - delta/2.0 * sx - A * sx / 2

    c_op_list = []
    c_op_list.append(sqrt(gamma1) * sm) # relaxation
    c_op_list.append(sqrt(gamma2) * sz) # dephasing
    
    start_time = time.time()
    output = mesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  
    expt_list4 = output.expect
    print 'Method 4: time elapsed = ' + str(time.time() - start_time)     
    
    # --------------------------------------------------------------------------
    # 5) Unitary evolution, constant hamiltonian
    #

    c_op_list = []
    
    start_time = time.time()
    output = mesolve(H_rwa, psi0, tlist, c_op_list, [sm.dag() * sm])  
    expt_list5 = output.expect
    print 'Method 5: time elapsed = ' + str(time.time() - start_time)         

    return expt_list1[0], expt_list2[0], expt_list3[0], expt_list4[0], expt_list5[0]
    
#
# set up the calculation
#
delta = 0.0 * 2 * pi   # qubit sigma_x coefficient
eps0  = 1.0 * 2 * pi   # qubit sigma_z coefficient
A     = 0.25 * 2 * pi  # driving amplitude (reduce to make the RWA more accurate)
w     = 1.0 * 2 * pi   # driving frequency
gamma1 = 0.2           # relaxation rate
gamma2 = 0.1           # dephasing  rate
psi0 = basis(2,1)      # initial state

tlist = linspace(0, 5.0 * 2 * pi / A, 250)

p_ex1, p_ex2, p_ex3, p_ex4, p_ex5 = qubit_integrate(delta, eps0, A, w, gamma1, gamma2, psi0, tlist)

plot(tlist, real(p_ex1), 'b', tlist, real(p_ex2), 'g.', tlist, real(p_ex3), 'r.-', tlist, real(p_ex4), 'c--', tlist, real(p_ex5), 'm')
xlabel('Time')
ylabel('Occupation probability')
title('Excitation probabilty of qubit')
legend(("Time-dependent function format", "Time-dependent string format", "Time-dependent func format", "Const. collapse operators", "Unitary evolution"))
show()


