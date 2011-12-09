#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from types import *
from scipy.integrate import *
from qutip.tidyup import tidyup
from qutip.Qobj import *
from qutip.superoperator import *
from qutip.expect import *
from qutip.Odeoptions import Odeoptions
from qutip.cyQ.ode_rhs import cyq_ode_rhs
from qutip.cyQ.codegen import Codegen
from qutip.Odedata import Odedata
import os,numpy

# ------------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
# 
def odesolve(H, rho0, tlist, c_op_list, expt_op_list, H_args=None, options=None):
    """
    Master equation evolution of a density matrix for a given Hamiltonian.

    Evolution of a state vector or density matrix (`rho0`) for a given
    Hamiltonian (`H`) and set of collapse operators (`c_op_list`), by integrating
    the set of ordinary differential equations that define the system. The
    output is either the state vector at arbitrary points in time (`tlist`), or
    the expectation values of the supplied operators (`expt_op_list`). 

    For problems with time-dependent Hamiltonians, `H` can be a callback function
    that takes two arguments, time and `H_args`, and returns the Hamiltonian
    at that point in time. `H_args` is a list of parameters that is
    passed to the callback function `H` (only used for time-dependent Hamiltonians).    
   
    Args:
    
        `H` (:class:`qutip.Qobj`): system Hamiltonian, or a callback function for time-dependent Hamiltonians.
        
        `rho0` (:class:`qutip.Qobj`): initial density matrix.
        
        `tlist` (*list/array*): list of times for :math:`t`.
        
        `c_op_list` (list of :class:`qutip.Qobj`): list of collapse operators.
        
        `expt_op_list` (list of :class:`qutip.Qobj`): list of operators for which to evaluate expectation values.
        
        `H_args` (*list*): of parameters to time-dependent Hamiltonians.
        
        `options` (:class:`qutip.Qdeoptions`): with options for the ODE solver.


    Returns:
    
        An *array* of expectation values of wavefunctions/density matrices
        for the times specified by `tlist`.

    .. note:: 
    
        On using callback function: odesolve transforms all :class:`qutip.Qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.Qobj` objects that are used in constructing the
        Hamiltonian via H_args. odesolve will check for :class:`qutip.Qobj` in
        `H_args` and handle the conversion to sparse matrices. All other
        :class:`qutip.Qobj` objects that are not passed via `H_args` will be
        passed on to the integrator to scipy who will raise an NotImplemented
        exception.

    """

    if options == None:
        options = Odeoptions()
        options.nsteps = 2500  # 

    if (c_op_list and len(c_op_list) > 0) or not isket(rho0):
        return me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list, H_args, options)
    else:
        return wf_ode_solve(H, rho0, tlist, expt_op_list, H_args, options)


# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution)
# 
def wf_ode_solve(H, psi0, tlist, expt_op_list, H_args, opt):
    """!
    Evolve the wave function using an ODE solver
    """
    if isinstance(H, list):
        return wf_ode_solve_td(H, psi0, tlist, expt_op_list, H_args, opt)

    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in range(n_tsteps)]
    else:
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # setup integrator
    #
    initial_vector = psi0.full()
    #r = scipy.integrate.ode(psi_ode_func)
    #r.set_f_params(-1.0j * H.data) # for python RHS
    r = scipy.integrate.ode(cyq_ode_rhs)
    L = -1.0j * H
    r.set_f_params(L.data.data, L.data.indices, L.data.indptr) # for cython RHS
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    psi = Qobj(psi0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        psi.data = r.y

        # calculate all the expectation values, or output psi if no operators where given
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(psi) # copy rho
        else:
            for m in range(0, n_expt_op):
                result_list[m,t_idx] = expect(expt_op_list[m], psi)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list

#
# evaluate dpsi(t)/dt
#
def psi_ode_func(t, psi, H):
    return H * psi

# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians
# 
def wf_ode_solve_td(H_func, psi0, tlist, expt_op_list,H_args, opt):
    """!
    Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """

    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in range(n_tsteps)]
    else:
        result_list=[]
        for i in xrange(n_expt_op):
            if expt_op_list[i].isherm:#preallocate real array of zeros
                result_list.append(zeros(n_tsteps))
            else:#preallocate complex array of zeros
                result_list.append(zeros(n_tsteps,dtype=complex))

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")
    #configure time-dependent terms
    if len(H_func)!=2:
        raise TypeError('Time-dependent Hamiltonian list must have two terms.')
    if (not isinstance(H_func[0],(list,ndarray))) or (len(H_func[0])<=1):
        raise TypeError('Time-dependent Hamiltonians must be a list with two or more terms')
    if (not isinstance(H_func[1],(list,ndarray))) or (len(H_func[1])!=(len(H_func[0])-1)):
        raise TypeError('Time-dependent coefficients must be list with length N-1 where N is the number of Hamiltonian terms.')
    tflag=1
    lenh=len(H_func[0])
    if opt.tidy:
        H_func[0]=[tidyup(H_func[0][k]) for k in range(lenh)]
    #create data arrays for time-dependent RHS function
    Hdata=[-1.0j*H_func[0][k].data.data for k in range(lenh)]
    Hinds=[H_func[0][k].data.indices for k in range(lenh)]
    Hptrs=[H_func[0][k].data.indptr for k in range(lenh)]
    #setup ode args string
    string=""
    for k in range(lenh):
        string+="Hdata["+str(k)+"],Hinds["+str(k)+"],Hptrs["+str(k)+"],"

    if H_args:
        td_consts=H_args.items()
        for elem in td_consts:
            string+=str(elem[1])
            if elem!=td_consts[-1]:
                string+=(",")
    #run code generator
    cgen=Codegen(lenh,H_func[1],H_args)
    cgen.generate("rhs.pyx")
    print 'Compiling...'
    os.environ['CFLAGS'] = '-w'
    import pyximport
    pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
    code = compile('from rhs import cyq_td_ode_rhs', '<string>', 'exec')
    exec(code)
    print 'Done.'
    #
    # setup integrator
    #

    initial_vector = psi0.full()
    r = scipy.integrate.ode(cyq_td_ode_rhs)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)                              
    r.set_initial_value(initial_vector, tlist[0])
    code = compile('r.set_f_params('+string+')', '<string>', 'exec')
    exec(code)

    # start evolution
    #
    psi = Qobj(psi0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        psi.data = r.y

        # calculate all the expectation values, or output psi if no operators where given
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(psi) # copy rho
        else:
            for m in range(0, n_expt_op):
                result_list[m][t_idx] = expect(expt_op_list[m], psi)

        r.integrate(r.t + dt)
        t_idx += 1
    os.remove("rhs.pyx")      
    return result_list



# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list, H_args, opt):
    """!
    Evolve the density matrix using an ODE solver
    """
    n_op= len(c_op_list)

    if isinstance(H, list):
        return me_ode_solve_td(H, rho0, tlist, c_op_list, expt_op_list, H_args, opt)

    if opt.tidy:
        H=tidyup(H,opt.atol)
    #
    # check initial state
    #
    if isket(rho0):
        # if initial state is a ket and no collapse operator where given,
        # fallback on the unitary schrodinger equation solver
        if n_op == 0:
            return wf_ode_solve(H, rho0, tlist, expt_op_list)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    # 
    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in range(n_tsteps)]
    else:
        result_list=[]
        for op in expt_op_list:
            if op.isherm and rho0.isherm:
                result_list.append(zeros(n_tsteps))
            else:
                result_list.append(zeros(n_tsteps,dtype=complex))

    #
    # construct liouvillian
    #
    L = liouvillian(H, c_op_list)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    #r = scipy.integrate.ode(rho_ode_func)
    #r.set_f_params(L.data)
    r = scipy.integrate.ode(cyq_ode_rhs)
    r.set_f_params(L.data.data, L.data.indices, L.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = vec2mat(r.y)
        
        # calculate all the expectation values, or output rho if no operators
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(rho) # copy rho
        else:
            for m in range(0, n_expt_op):
                result_list[m][t_idx] = expect(expt_op_list[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list

#
# evaluate drho(t)/dt according to the master eqaution
#
def rho_ode_func(t, rho, L):
    return L*rho

# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_solve_td(H_func, rho0, tlist, c_op_list, expt_op_list, H_args, opt):
    """!
    Evolve the density matrix using an ODE solver with time dependent
    Hamiltonian.
    """
    n_op= len(c_op_list)

    #
    # check initial state
    #
    if isket(rho0):
        # if initial state is a ket and no collapse operator where given,
        # fallback on the unitary schrodinger equation solver
        if n_op == 0:
            return wf_ode_solve_td(H_func, rho0, tlist, expt_op_list, H_args, opt)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    # 
    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in range(n_tsteps)]
    else:
        result_list=[]
        for op in expt_op_list:
            if op.isherm:
                result_list.append(zeros(n_tsteps))
            else:
                result_list.append(zeros(n_tsteps),dtype=complex)

    #
    # construct liouvillian
    #
    if len(H_func)!=2:
        raise TypeError('Time-dependent Hamiltonian list must have two terms.')
    if (not isinstance(H_func[0],(list,ndarray))) or (len(H_func[0])<=1):
        raise TypeError('Time-dependent Hamiltonians must be a list with two or more terms')
    if (not isinstance(H_func[1],(list,ndarray))) or (len(H_func[1])!=(len(H_func[0])-1)):
        raise TypeError('Time-dependent coefficients must be list with length N-1 where N is the number of Hamiltonian terms.')
    
    lenh=len(H_func[0])
    if opt.tidy:
        H_func[0]=[tidyup(H_func[0][k]) for k in range(lenh)]
    L_func=[[liouvillian(H_func[0][0], c_op_list)],H_func[1]]
    for m in range(1, lenh):
        L_func[0].append(liouvillian(H_func[0][m],[]))

    #create data arrays for time-dependent RHS function
    Ldata=[L_func[0][k].data.data for k in range(lenh)]
    Linds=[L_func[0][k].data.indices for k in range(lenh)]
    Lptrs=[L_func[0][k].data.indptr for k in range(lenh)]
    #setup ode args string
    string=""
    for k in range(lenh):
        string+="Ldata["+str(k)+"],Linds["+str(k)+"],Lptrs["+str(k)+"],"
    if H_args:
        td_consts=H_args.items()
        for elem in td_consts:
            string+=str(elem[1])
            if elem!=td_consts[-1]:
                string+=(",")
    
    #run code generator
    cgen=Codegen(lenh,L_func[1],H_args)
    cgen.generate("rhs.pyx")
    print 'Compiling...'
    os.environ['CFLAGS'] = '-w'
    import pyximport
    pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
    code = compile('from rhs import cyq_td_ode_rhs', '<string>', 'exec')
    exec(code)
    print 'Done.'
    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cyq_td_ode_rhs)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                              first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    code = compile('r.set_f_params('+string+')', '<string>', 'exec')
    exec(code)

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = vec2mat(r.y)

        # calculate all the expectation values, or output rho if no operators
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(rho) # copy rho
        else:
            for m in range(0, n_expt_op):
                result_list[m][t_idx] = expect(expt_op_list[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
    os.remove("rhs.pyx")      
    return result_list





