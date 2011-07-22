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
###########################################################################

import random
from types import *
from scipy.integrate import *

from Qobj import *
from superoperator import *
from expect import *
from Odeoptions import Odeoptions

# ------------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
# 
def odesolve(H, rho0, tlist, c_op_list, expt_op_list, H_args=None):
    """
    Evolution of a state vector or density matrix (rho0) for a given
    Hamiltonian (H) and set of collapse operators (c_op_list), by integrating
    the set of ordinary differential equations that define the system. The
    output is either the state vector at arbitrary points in time (tlist), or
    the expectation values of the supplied operators (expt_op_list). 

    For problems with time-dependent Hamiltonians, H can be a callback function
    that takes two arguments, time and H_args, and returns the Hamiltonian
    at that point in time. H_args is a list of parameters that is
    passed to the callback function H (only used for time-dependent Hamiltonians).
    """

    if c_op_list and len(c_op_list) > 0:
        return me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list, H_args)
    else:
        return wf_ode_solve(H, rho0, tlist, expt_op_list, H_args)


# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution)
# 
def wf_ode_solve(H, psi0, tlist, expt_op_list, H_args=None):
    """!
    @brief Evolve the wave function using an ODE solver
    """
    if isinstance(H, FunctionType):
        return wf_ode_solve_td(H, psi0, tlist, expt_op_list, H_args)

    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in xrange(n_tsteps)]
    else:
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    if not isket(psi0):
        raise TypeError("psi0 must be a ket")

    #
    # setup integrator
    #
    initial_vector = psi0.full()
    r = scipy.integrate.ode(psi_ode_func)
    opt = Odeoptions()
    r.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step)
    #r.set_integrator('zvode')
    r.set_initial_value(initial_vector, tlist[0]).set_f_params(-1.0j * H.data)

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
def wf_ode_solve_td(H_func, psi0, tlist, expt_op_list, H_args):
    """!
    @brief Evolve the wave function using an ODE solver with time-dependent
    Hamiltonian.
    """

    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in xrange(n_tsteps)]
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
    r = scipy.integrate.ode(psi_ode_func_td).set_integrator('zvode').set_initial_value(initial_vector, tlist[0]).set_f_params(H_func_and_args)

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
            for m in xrange(0, n_expt_op):
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
def me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list, H_args=None):
    """!
    @brief Evolve the density matrix using an ODE solver
    """
    n_op= len(c_op_list)

    if isinstance(H, FunctionType):
        return me_ode_solve_td(H, rho0, tlist, c_op_list, expt_op_list, H_args)

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
        result_list = [Qobj() for k in xrange(n_tsteps)]
    else:
        result_list=[]
        for op in expt_op_list:
            if op.isherm:
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
    initial_vector = rho0.full().reshape(prod(rho0.shape),1)
    r = scipy.integrate.ode(rho_ode_func).set_integrator('zvode').set_initial_value(initial_vector, tlist[0]).set_f_params(L.data)

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = r.y.reshape(rho0.shape)

        # calculate all the expectation values, or output rho if no operators
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(rho) # copy rho
        else:
            for m in xrange(0, n_expt_op):
                result_list[m][t_idx] = expect(expt_op_list[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list


#
# evaluate drho(t)/dt according to the master eqaution
#
def rho_ode_func(t, rho, L):
    return L * rho



# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_solve_td(H_func, rho0, tlist, c_op_list, expt_op_list, H_args):
    """!
    @brief Evolve the density matrix using an ODE solver with time dependent
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
            return wf_ode_solve_td(H_func, rho0, tlist, expt_op_list, H_args)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    # 
    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]

    if n_expt_op == 0:
        result_list = [Qobj() for k in xrange(n_tsteps)]
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
    for m in xrange(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

    L_func_and_args = [H_func, L.data]
    for arg in H_args:
        if isinstance(arg,Qobj):
            L_func_and_args.append((-1j*(spre(arg) - spost(arg))).data) # hack: assume all Qobj to be hamiltonian components....
        else:
            L_func_and_args.append(arg)

    #
    # setup integrator
    #
    initial_vector = rho0.full().reshape(prod(rho0.shape),1)
    r = scipy.integrate.ode(rho_ode_func_td).set_integrator('zvode').set_initial_value(initial_vector, tlist[0]).set_f_params(L_func_and_args)

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = r.y.reshape(rho0.shape)

        # calculate all the expectation values, or output rho if no operators
        if n_expt_op == 0:
            result_list[t_idx] = Qobj(rho) # copy rho
        else:
            for m in xrange(0, n_expt_op):
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













