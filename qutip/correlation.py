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

from scipy import *
from Qobj import *
from spre import *
from spost import *
from Counter import *
from ode_solve import *
from ode2es import *
from mcsolve import *

def correlation_es(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function <A(t+tau)B(t)> using exponential
    series and the quantum regression theorem.
    """

    # contruct the Liouvillian
    n_op      = len(c_op_list)
    L = -1.0j * (spre(H) - spost(H))
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

    if rho0 == None:
        rho0 = steady(L)

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    pgb = Counter(len(tlist), "Correlation function")

    solES_t = ode2es(L, rho0)

    for t_idx in range(len(tlist)):

        rho_t = esval_op(solES_t, [tlist[t_idx]])

        solES_tau = ode2es(L, b_op * rho_t)

        C_mat[t_idx, :] = esval(scalar_expect(a_op, solES_tau), taulist)
   
        pgb.update()

    pgb.finish()

    return C_mat


def correlation_ode(H, rho0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function <A(t+tau)B(t)> using the ode
    solver, and the quantum regression theorem.
    """

    if rho0 == None:
        # contruct the Liouvillian
        n_op      = len(c_op_list)
        L = -1.0j * (spre(H) - spost(H))
        for m in range(0, n_op):
            cdc = c_op_list[m].dag() * c_op_list[m]
            L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

        rho0 = steady(L)

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    pgb = Counter(len(tlist), "Correlation function")

    rho_t = me_ode_solve(H, rho0, tlist, c_op_list, [])

    for t_idx in range(len(tlist)):

        C_mat[t_idx, :] = me_ode_solve(H, b_op * rho_t[t_idx], taulist, c_op_list, [a_op])  
  
        pgb.update()

    pgb.finish()

    return C_mat



def correlation_mc(H, psi0, tlist, taulist, c_op_list, a_op, b_op):
    """
    Calculate a two-time correlation function <A(t+tau)B(t)> using the
    Monte-Carle solver, and the quantum regression theorem.
    """

    C_mat = zeros([size(tlist),size(taulist)],dtype=complex)

    pgb = Counter(len(tlist), "Correlation function")

    ntraj = 20

    mc_opt = Mcoptions()
    mc_opt.progressbar = False

    psi_t, t_collapse = mcsolve(H, psi0, tlist, ntraj, c_op_list, [], mc_opt)

    for t_idx in range(len(tlist)):

        psi0_t = psi_t[0][t_idx]

        ops = mcsolve(H, b_op * psi0_t, tlist, ntraj, c_op_list, [a_op], mc_opt)
        avg = sum(ops, axis=0) / ntraj
        C_mat[t_idx, :] = avg
  
        pgb.update()

    pgb.finish()

    return C_mat



