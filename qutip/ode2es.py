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
from qobj import *
from eseries import *

#
# Create an exponential series that describes the time evolution for the initial
# state rho0, given the Liouvillian L.
#
def ode2es(L, rho0):

    # todo: some sanity test... check that L issuper and rho isoper, convert
    # rho0 to operator if it isket
 
    w, v = la.eig(L.full())
    # w[i]   = eigenvalue i
    # v[:,i] = eigenvector i

    rlen = prod(rho0.shape)
    r0 = reshape(rho0.full(),    [rlen, 1])
    v0 = la.solve(v,r0)
    vv = v * sp.spdiags(v0.T, 0, rlen, rlen)

    out = None
    for i in range(rlen):
        qo = qobj(reshape(vv[:,i], rho0.shape), dims=rho0.dims, shape=rho0.shape)
        if out:
            out += eseries(qo, w[i])
        else:
            out  = eseries(qo, w[i])

    return estidy(out)
