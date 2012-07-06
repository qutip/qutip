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

from qutip.tensor import tensor
from qutip.superoperator import spre, spost, mat2vec, vec2mat
from qutip.Qobj import Qobj
from numpy import hstack
import scipy.linalg as la
from qutip.graph import matrix_histogram, matrix_histogram_complex

from pylab import *

def index_permutations(size_list, perm=[]):
    """
    Generate a list with all index permutations. size_list is a list that 
    contains the sizes for each composite system.
    """
    if len(size_list) == 0:
        yield perm
    else:
        for n in range(size_list[0]):
            for ip in index_permutations(size_list[1:], perm + [n]):
                yield ip

def qpt_plot(chi, lbls_list, title=None, fig=None):
    """
    Visualize the quantum process tomography chi matrix. Plot the real and
    imaginary separately.
    """
    if fig == None:
        fig = figure(figsize=(16,8))

    xlabels = []
    for inds in index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]] for k in range(len(lbls_list))]))        

    ax = None

    ax = fig.add_subplot(1,2,1, projection='3d')
    matrix_histogram(real(chi), xlabels, xlabels, r"real($\chi$)", [-1,1], ax)

    ax = fig.add_subplot(1,2,2, projection='3d')
    matrix_histogram(imag(chi), xlabels, xlabels, r"imag($\chi$)", [-1,1], ax)

    if title:
        fig.suptitle(title)

def qpt_plot_complex(chi, lbls_list, title=None, fig=None):
    """
    Visualize the quantum process tomography chi matrix. Plot bars with
    height that correspond to the absolute value and color that correspond
    to the phase.
    """
    #if fig == None:
    #    fig = figure(figsize=(16,8))

    xlabels = []
    for inds in index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]] for k in range(len(lbls_list))]))        

    # use given fig instance...
    
    if not title:
        title = r"$\chi$"

    matrix_histogram_complex(chi, xlabels, xlabels, title)

def qpt(U, op_basis_list):
    """
    Calculate the quantum process tomography chi matrix for a given 
    (possibly nonunitary) transformation matrix U, which transforms a 
    density matrix in vector form according to:

        vec(rho) = U * vec(rho0)

        or

        rho = vec2mat(U * mat2vec(rho0))

    U can be calculated for an open quantum system using the QuTiP propagator
    function.
    """

    E_ops = []
    # loop over all index permutations
    for inds in index_permutations([len(op_list) for op_list in op_basis_list]):
        # loop over all composite systems
        E_op_list = [op_basis_list[k][inds[k]] for k in range(len(op_basis_list))]
        E_ops.append(tensor(E_op_list))

    EE_ops = [spre(E1) * spost(E2.dag()) for E1 in E_ops for E2 in E_ops]

    M = hstack([mat2vec(EE.full()) for EE in EE_ops])

    Uvec = mat2vec(U.full())

    chi_vec = la.solve(M, Uvec)

    return vec2mat(chi_vec)