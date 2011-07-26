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
from scipy import *
from Qobj import *
from eseries import *
from expect import *
from superoperator import *


# ------------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
# 
def essolve(H, rho0, tlist, c_op_list, expt_op_list):
    """
    Evolution of a state vector or density matrix (rho0) for a given
    Hamiltonian (H) and set of collapse operators (c_op_list), by expressing
    the ODE as an exponential series. 

    The output is either the state vector at arbitrary points in time (tlist),
    or the expectation values of the supplied operators (expt_op_list). 

    This solver does not support time-dependent Hamiltonians.
    """
    n_expt_op = len(expt_op_list)
    n_tsteps  = len(tlist)

    # check initial state
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    # Calculate the Liouvillian
    L = liouvillian(H, c_op_list)

    # calculate the exponential series
    es = ode2es(L, rho0)

    # evaluate the expectation values      
    if n_expt_op == 0:
        result_list = [Qobj() for k in xrange(n_tsteps)]
    else:
        result_list = zeros([n_expt_op, n_tsteps], dtype=complex)

    for n in range(0, n_expt_op):
        result_list[n,:] = esval(expect(expt_op_list[n],es),tlist)

    return result_list

# ------------------------------------------------------------------------------
#
#
def ode2es(L, rho0):
    """
    Create an exponential series that describes the time evolution for the
    initial state rho0, given the Liouvillian L.
    """

    # todo: some sanity test... check that L issuper and rho isoper, convert
    # rho0 to operator if it isket
 
    w, v = la.eig(L.full())
    # w[i]   = eigenvalue i
    # v[:,i] = eigenvector i

    rlen = prod(rho0.shape)
    r0 = mat2vec(rho0.full())
    v0 = la.solve(v,r0)
    vv = v * sp.spdiags(v0.T, 0, rlen, rlen)

    out = None
    for i in range(rlen):
        qo = Qobj(vec2mat(vv[:,i]), dims=rho0.dims, shape=rho0.shape)
        if out:
            out += eseries(qo, w[i])
        else:
            out  = eseries(qo, w[i])

    return estidy(out)
