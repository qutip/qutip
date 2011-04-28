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

from scipy.integrate import *

from Qobj import *
from spre import *
from spost import *
from expect import *

# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution)
# 
def wf_ode_solve(H, psi0, tlist, expt_op_list):
    """!
    @brief Evolve the wave function using an ODE solver
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
    initial_vector = psi0.full()
    r = scipy.integrate.ode(psi_ode_func).set_integrator('zvode').set_initial_value(initial_vector, tlist[0]).set_f_params(H.data)

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
    return -1j * (H * psi)


# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list):
    """!
    @brief Evolve the density matrix using an ODE solver
    """
    n_op      = len(c_op_list)

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
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    #
    # construct liouvillian
    #
    L = -1j*(spre(H) - spost(H))
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

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
            for m in range(0, n_expt_op):
                result_list[m,t_idx] = expect(expt_op_list[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list


#
# evaluate drho(t)/dt according to the master eqaution
#
def rho_ode_func(t, rho, L):
    return L * rho
















