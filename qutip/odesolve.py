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
from qutip.rhs_generate import rhs_generate
from qutip.Odedata import Odedata
import os,numpy,odeconfig


#class QobjTD:
#    """
#    Internal class for management of time-dependent quantum objects:
#    
#    Is it useful with a class for this? too large overhead?
#   """
#    
#    def __init__(self, qobj_spec, args):
#       #
#        # qobj_spec can be in one of the following formats:
#        #
#        #   * Qobj : time-independent
#        #   * f(t) : Qobj(t) = f(t, args) # here args contains Qobjs
#        #   * [Qobj, f(t)]  : Qobj(t) = Qobj * f(t, args)
#        #   * [Qobj, 'str'] : Qobj(t) = Qobj * eval(str @ t using args)
#        #

# ------------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
# 
def mesolve(H, rho0, tlist, c_ops, expt_ops, args={}, options=None):
    """
    New master equation API: still a moving target...
    
    """

    if options == None:
        options = Odeoptions()
        options.nsteps = 2500  #
        
    # do some basic sanity checks on the format of H and c_ops: this should
    # be moved to a function that can be reused.
    n_const = 0
    n_func  = 0
    n_str   = 0
    
    # check H for incorrect format
    if isinstance(H, Qobj):    
        n_const += 1
    elif isinstance(H, FunctionType):
        n_func += 1
    elif isinstance(H, list):
        for h in H:
            if isinstance(h, Qobj):
                n_const += 1
            elif isinstance(h, list):
                if len(h) != 2 or not isinstance(h[0], Qobj):
                    raise TypeError("Incorrect hamiltonian specification")
                else:
                    if isinstance(h[1], FunctionType):
                        n_func += 1
                    else isinstance(h[1], str):
                        n_str += 1
                    else:
                        raise TypeError("Incorrect hamiltonian specification")
    else:
        raise TypeError("Incorrect hamiltonian specification")
        
    # the the whole thing again for c_ops 
    if isinstance(c_ops, list):
        for c in c_ops:
            if isinstance(c, Qobj):
                n_const += 1
            elif isinstance(c, list):
                if len(c) != 2 or not isinstance(c[0], Qobj):
                    raise TypeError("Incorrect collapse operator specification")
                else:
                    if isinstance(c[1], FunctionType):
                        n_func += 1
                    else isinstance(c[1], str):
                        n_str += 1
                    else:
                        raise TypeError("Incorrect collapse operator specification")
    else:
        raise TypeError("Incorrect collapse operator specification")       

    #
    # if n_str == 0 and n_func == 0:
    #     # no time-dependence at all
    #
    if n_str > 0 and n_func > 0:
        raise TypeError("Cannot mix string and function type time-dependence formats")       


    #
    #
    #         
        
    if (c_ops and len(c_ops) > 0) or not isket(rho0):
        #
        # we have collapse operators
        #
        
        #
        # find out if we are dealing with all-constant hamiltonian and 
        # collapse operators or if we have at least one time-dependent
        # operator. Then delegate to appropriate solver...
        #
                
        if isinstance(H, Qobj):
            # constant hamiltonian
            raise TypeError("Not Implemented: use old API for now")           
        
        if isinstance(H, FunctionType):
            # old style time-dependence
            raise TypeError("Not Implemented: use old API for now")
        
        if isinstance(H, list):
            # determine if we are dealing with list of [Qobj, string] or [Qobj, function]
            # style time-dependences (for pure python and cython, respectively)
            return mesolve_list_td(H, rho0, tlist, c_ops, expt_ops, args, options)
                       
        raise TypeError("Unknown parameter types")

    else:
        #
        # no collapse operators: unitary dynamics
        #
        raise TypeError("Not Implemented: use old API for now")


# ------------------------------------------------------------------------------
# A time-dependent disipative master equation on the list form
# 
def mesolve_list_td(H_list, rho0, tlist, c_list, expt_ops, args, opt):
    """
    New master equation APIs: still a moving target...    
    """
    n_op= len(c_list)

    #
    # check initial state
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    # 
    n_expt_op = len(expt_ops)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in range(n_tsteps)]
    else:
        result_list=[]
        for op in expt_ops:
            if op.isherm:
                result_list.append(zeros(n_tsteps))
            else:
                result_list.append(zeros(n_tsteps),dtype=complex)

    #
    # construct liouvillian in list-function format
    #
    L_list = []
    constant_func = lambda x,y: 1.0;
    
    # add all hamitonian terms to the lagrangian list
    for h_spec in H_list:
    
        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = constant_func
   
        elif isinstance(h_spec, list): 
            h = h_spec[0]
            h_coeff = h_spec[1]
            
        else:
            raise TypeError("Incorrect specification of time-dependent Hamiltonian")
                
        L_list.append([(-1j*(spre(h) - spost(h))).data, h_coeff])
        
        
    # add all collapse operators to the lagrangian list    
    for c_spec in c_list:

        if isinstance(c_spec, Qobj):
            c = c_spec
            c_coeff = constant_func
   
        elif isinstance(c_spec, list): 
            c = c_spec[0]
            c_coeff = c_spec[1]
            
        else:
            raise TypeError("Incorrect specification of time-dependent collapse operators")
                
        cdc = c.dag() * c
        L_list.append([(spre(c)*spost(c.dag())-0.5*spre(cdc)-0.5*spost(cdc)).data, c_coeff])    
        
    L_list_and_args = [L_list, args]

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(rho_list_td)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                              first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_list_and_args)

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
                result_list[m][t_idx] = expect(expt_ops[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list


#
# evaluate drho(t)/dt according to the master equation using the [Qobj, function]
# style time dependence API
#
def rho_list_td(t, rho, L_list_and_args):

    L_list = L_list_and_args[0]
    args   = L_list_and_args[1]

    L = L_list[0][0] * L_list[0][1](t, args)
    for n in range(1,len(L_list)):
        #
        # L_args[n][0] = the sparse data for a Qobj in super-operator form
        # L_args[n][1] = function callback giving the coefficient
        #
        L = L + L_list[n][0] * L_list[n][1](t, args)
    
    return L * rho

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
        
        `c_ops` (list of :class:`qutip.Qobj`): list of collapse operators.
        
        `expt_ops` (list of :class:`qutip.Qobj`): list of operators for which to evaluate expectation values.
        
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
    if isinstance(H, FunctionType):
        return wf_ode_solve_func_td(H, psi0, tlist, expt_op_list, H_args, opt)

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
    if opt.rhs_reuse==True and odeconfig.tdfunc==None:
        print "No previous time-dependent RHS found."
        print "Generating one for you..."
        rhs_generate(H_func,H_args)
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
    if not opt.rhs_reuse:
        name="rhs"+str(odeconfig.cgen_num)
        odeconfig.tdname=name
        cgen=Codegen(lenh,H_func[1],H_args)
        cgen.generate(name+".pyx")
        print "Compiling '"+name+".pyx' ..."
        os.environ['CFLAGS'] = '-w'
        import pyximport
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
        code = compile('from '+name+' import cyq_td_ode_rhs', '<string>', 'exec')
        exec(code)
        print 'Done.'
        odeconfig.tdfunc=cyq_td_ode_rhs
    #
    # setup integrator
    #

    initial_vector = psi0.full()
    r = scipy.integrate.ode(odeconfig.tdfunc)
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
    if not opt.rhs_reuse:
        os.remove(name+".pyx")      
    return result_list

# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution), for
# time dependent hamiltonians
# 
def wf_ode_solve_func_td(H_func, psi0, tlist, expt_op_list, H_args, opt):
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
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # setup integrator
    #
    H_func_and_args = [H_func]
    for arg in H_args:
        if isinstance(arg,Qobj):
            H_func_and_args.append(arg.data)
        else:
            H_func_and_args.append(arg)

    initial_vector = psi0.full()
    r = scipy.integrate.ode(psi_ode_func_td)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)                              
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(H_func_and_args)

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
# evaluate dpsi(t)/dt for time-dependent hamiltonian
#
def psi_ode_func_td(t, psi, H_func_and_args):
    H_func = H_func_and_args[0]
    H_args = H_func_and_args[1:]

    H = H_func(t, H_args)

    return -1j * (H * psi)

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

    if isinstance(H, FunctionType):
        return me_ode_solve_func_td(H, rho0, tlist, c_op_list, expt_op_list, H_args, opt)


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
    if opt.rhs_reuse==True and odeconfig.tdfunc==None:
        print "No previous time-dependent RHS found."
        print "Generating one for you..."
        rhs_generate(H_func,H_args)
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
    if not opt.rhs_reuse:
        name="rhs"+str(odeconfig.cgen_num)
        cgen=Codegen(lenh,L_func[1],H_args)
        cgen.generate(name+".pyx")
        print "Compiling '"+name+".pyx' ..."
        os.environ['CFLAGS'] = '-w'
        import pyximport
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
        code = compile('from '+name+' import cyq_td_ode_rhs', '<string>', 'exec')
        exec(code)
        print 'Done.'
        odeconfig.tdfunc=cyq_td_ode_rhs
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(odeconfig.tdfunc)
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
    if not opt.rhs_reuse:
        os.remove(name+".pyx")      
    return result_list


# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_solve_func_td(H_func, rho0, tlist, c_op_list, expt_op_list, H_args, opt):
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
    L = 0
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        if L == 0:
            L = spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)            
        else:
            L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

    L_func_and_args = [H_func, L.data]
    for arg in H_args:
        if isinstance(arg,Qobj):
            L_func_and_args.append((-1j*(spre(arg) - spost(arg))).data)
        else:
            L_func_and_args.append(arg)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(rho_ode_func_td)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                              first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_func_and_args)

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
def rho_ode_func_td(t, rho, L_func_and_args):

    L_func = L_func_and_args[0]
    L0     = L_func_and_args[1]
    L_args = L_func_and_args[2:]

    L = L0 + L_func(t, L_args)

    return L * rho



