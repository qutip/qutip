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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from qobj import *
from scipy.integrate import *

import random

# ------------------------------------------------------------------------------
# Wave function evolution using a ODE solver (unitary quantum evolution)
# 
def wf_ode_evolve(tlist, H, psi0, expt_op_list):
    """!
    @brief Evolve the wave function using an ODE solver
    """
    n_tsteps  = len(tlist)
    n_expt_op = len(expt_op_list)
    expt_list = zeros([n_expt_op, n_tsteps])
    dt        = tlist[1]-tlist[0]

    r = scipy.integrate.ode(mcwf_ode_func)

    initial_vector = psi0.full() 
    #print "initial vector = ", initial_vector

    r.set_integrator('zvode').set_initial_value(initial_vector, 0.0).set_f_params(H)

    t_idx = 0
    for m in range(0, n_expt_op):
        expt_list[m,t_idx] += expect(expt_op_list[m], psi0)
    
    while r.successful() and r.t < tlist[-1]:
        t_idx += 1

        psi = qobj(r.y)
        psi.dims = psi0.dims

        # calculate all the expectation values: contribution from single trajectory
        for m in range(0, n_expt_op):
            expt_list[m,t_idx] += expect(expt_op_list[m], psi)

        r.integrate(r.t + dt)
           
    return expt_list

#
# evaluate dpsi(t)/dt
#
def mcwf_ode_func(t, psi, H):
    dpsi = -1j * (H.data * psi)
    return dpsi


# ------------------------------------------------------------------------------
# Master equation solver
# 
def me_ode_evolve(tlist, H, rho0, c_op_list, expt_op_list):
    """!
    @brief Evolve the denstiy matrix using an ODE solver
    """
    n_tsteps  = len(tlist)
    n_expt_op = len(expt_op_list)
    n_op      = len(c_op_list)
    expt_list = zeros([n_expt_op, n_tsteps], dtype=complex)
    dt        = tlist[1]-tlist[0]

    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * trans(rho0)

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
    r = scipy.integrate.ode(me_ode_func).set_integrator('zvode').set_initial_value(initial_vector, tlist[0]).set_f_params(L.data)

    #r.set_integrator('zvode').set_initial_value(initial_vector, 0.0).set_f_params(L)

    #
    # start evolution
    #
    rho = qobj(rho0)

    t_idx = 0
    # while t_idx <= n_tsteps:
    for t in tlist:
        if not r.successful():
            break;

        rho.data = r.y.reshape(rho0.shape)

        # calculate all the expectation values
        for m in range(0, n_expt_op):
            expt_list[m,t_idx] = expect(expt_op_list[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return expt_list


#
# evaluate drho(t)/dt according to the master eqaution
#
def me_ode_func(t, rho, L):
    drho = (L * rho)
    return drho

















