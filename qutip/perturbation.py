# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Xiang Gao.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['Perturbation','EnergyLevelPerturbation','PerturbedEigenSpace','PerturbedBase']

from qutip import Qobj
from numpy import zeros, linalg

# For formula used in this module, and the explanation of data structure of this module,
# see the ipython notebook "example-perturbation.ipynb". Note that the source code and
# comments in this module should always be kept consistent with that note book. If you
# make changes to this module, ALWAYS update "example-perturbation.ipynb" - and vice versa.
# If you have questions about this module, feel free to email qasdfgtyuiop@gmail.com

class EnergyLevelPerturbation:
    """This class is the node of energy level forest. For more detail, see
    section 2.1 in example-perturbation.ipynb
        
    """
        
    def __init__(self, energy, degeneracy, parent = None, children = None, eigen_space = None):
        
        # value of energy level perturbation
        self.energy = energy
        
        # degeneracy of this energy level perturbation
        self.degeneracy = degeneracy
        
        # the eigen space corresponding to this energy level perturbation
        # if this energy level is splitted in furter perturbation, then the
        # value of eigen_space will be None
        self.eigen_space = eigen_space

        # list of pointers to the children
        self.children = children
        if children is None:
            self.children = []

        # pointer to the previous order energy level perturbation
        self.parent = parent

class PerturbedBase:
    """This class stores a state vector and its perturbation in a list, see
    section 2.2 in example-perturbation.ipynb for more detail.
    """

    def __init__(self,vectors=None):
        self.vectors = vectors
        if vectors is None:
            self.vectors = []

    # overload of arithmethic operators
    def __add__(self,other):
        if isinstance(other,int):
            # in order to be able to use sum()
            return self*1
        else:
            return PerturbedBase([x + y for x, y in zip(self.vectors, other.vectors)])

    def __radd__(self,other):
        if isinstance(other,int):
            # in order to be able to use sum()
            return self*1
        else:
            return PerturbedBase([x + y for x, y in zip(other.vectors, self.vectors)])

    def __sub__(self,other):
        return PerturbedBase([x - y for x, y in zip(self.vectors, other.vectors)])

    def __mul__(self,num):
        return PerturbedBase([x * num for x in self.vectors])

    def __rmul__(self,num):
        return self*num

    def __div__(self,num):
        return self*(1.0/num)

    def __neg__(self):
        return self*(-1)

    def __pos__(self):
        return self*1


class PerturbedEigenSpace:
    """The instance of this class stores the information of the "most splitted
    subspace" (defined in section 2 of example-perturbation.ipynb). See section 2.3
    of example-perturbation.ipynb for further information.
        
    """

    def __init__(self, base_set=None, elp_node=None):
        self.elp_node = elp_node
        self.base_set = base_set
        if base_set is None:
            self.base_set = []

    def split(self,eigen_system_info):
        """See section 1.2.2 in example-perturbation.ipynb, given the
        result for the eigenvalue problem (1.2.2.4), split this eigenspace
        into several subspaces, update the corresponding structure in
        energy level forest and corresponding references.

        Parameters
        ----------
        eigen_system_info : array
            classified eigenvalues and eigenvectors of problem (1.2.2.4)
            according to degenerate. it is such a data structure:
            [ [eigen value 1,[eigen vector 1, eigen vector 2, ...]],
              [eigen value 2,[eigen vector 1, eigen vector 2, ...]],
              ...  ]

        Returns
        -------
        [PerturbedEigenSpace]
            list of newly splitted subspace
        """
        ret = []
        self.elp_node.eigen_space = None
        for i in eigen_system_info: # for each newly splitted subspace
            # retrieve information about energy level perturbation
            energy = i[0]
            degeneracy = len(i[1])
            # calculate new base set for newly splitted subspace
            new_base_set = [ sum([x * y for x,y in zip(j,self.base_set)]) for j in i[1] ]
            # create a new instance of newly splitted subspace
            space = PerturbedEigenSpace(new_base_set)
            ret.append(space)
            # create new nodes in energy level tree
            node = EnergyLevelPerturbation(energy, degeneracy, self.elp_node, None, space)
            self.elp_node.children.append(node)
            space.elp_node = node
        return ret

    def energy_levels(self):
        """Returns the list of energy corrections of different order perturbations
        corresponding to this subspace, i.e. the corresponding [E^(0),E^(1),...]
        
        Returns
        -------
        [EnergyLevelPerturbation]
            the list of energy corrections of different order perturbations
            corresponding to this subspace, i.e. the corresponding [E^(0),E^(1),...]
        """
        node = self.elp_node
        ret = [node]
        while not (node.parent is None):
            node = node.parent
            ret = [node] + ret
        return ret


class Perturbation:
    """This class is the calculator class that do the calculation
    of perturbation theory. See section 2.4 for detailed information.
        
    """
    
    def categorize_eigenstates(self,eigenvalues,eigenstates):
        """This is an inner method used by other methods of this class,
        it is not designed for user's usage.
        
        it categorize eigenvalues and eigenvectors by the degeneracy
        here we maintain a data structure:
        [ [eigen value 1,[eigen vector 1, eigen vector 2, ...]],
          [eigen value 2,[eigen vector 1, eigen vector 2, ...]],
          ...  ]
            
        Parameters
        ----------
        eigenvalues : array
            the eigenvalues
        eigenstates : array
            the array of eigenstates
        """
        eigen_system_info = []
        for i in range(len(eigenvalues)):
            eigen_value = eigenvalues[i]
            eigen_vector = eigenstates[i]
            for j in eigen_system_info:
                if abs(j[0] - eigen_value) < self.etol:
                    j[1] += [eigen_vector]
                    break
            else:
                # not degenerate
                eigen_system_info += [[eigen_value, [eigen_vector]]]
        return eigen_system_info

    def __init__(self, h0, zero_order_eigenstates=None,
                 zero_order_energy_levels=None, etol=1E-8):
        """Initialize the calculator class with H0 and its eigen values and vectors

        Parameters
        ----------
        h0 : Qobj
            the unperturbed Hamiltonian H0
        zero_order_eigenstates : [Qobj]
            list of eigenstates of H0. this parameter has a default
            value None. if this value is None, then the build-in solver
            of qutip will be automatically called to solve for the eigen
            states and the parameter zero_order_energy_levels will be ignored.
        zero_order_energy_levels : [real numbers]
            list of energy levels of H0 corresponding to each value in
            zero_order_eigenstates. this parameter has a default value
            None. if it's None but zero_order_eigenstates is not None,
            it will be automatically calculated by E = <phi|H0|phi>/<phi|phi>
        etol : float
            tolerance of energy, i.e. if |Em-En|<etol, we regard
            the two energy are degenerate
        """
        # all the n^th order perturbation Hamiltonian
        self.hamiltonians = [h0]
        # the order of perturbation that we have calculated.
        self.order = 0
        # tolerance of energy
        self.etol = etol
        
        # if zero_order_eigenstates or zero_order_energy_levels are not
        # set, calculate them
        if zero_order_eigenstates is None:
            zero_order_energy_levels, zero_order_eigenstates = h0.eigenstates()
        if zero_order_energy_levels is None:
            zero_order_energy_levels = [(x.dag()*h0*x).tr()/(x.dag()*x).tr() for x in zero_order_eigenstates]
        
        # categorize eigenvalues and eigenvectors.
        eigen_system_info = self.categorize_eigenstates(zero_order_energy_levels,zero_order_eigenstates)
        
        # build from categorized eigen system information
        self.energy_level_trees = []
        self.eigen_spaces = []
        for i in eigen_system_info: # for each newly splitted subspace
            # retrieve information about energy level perturbation
            energy = i[0]
            eigenvectors = i[1]
            degeneracy = len(i[1])
            # calculate new base set for newly splitted subspace
            base_set = [ PerturbedBase([j]) for j in i[1] ]
            # create a new instance of newly splitted subspace
            space = PerturbedEigenSpace(base_set)
            self.eigen_spaces.append(space)
            # create new nodes in energy level tree
            node = EnergyLevelPerturbation(energy, degeneracy, None, None, space)
            self.energy_level_trees.append(node)
            space.elp_node = node

    def next_order(self, ht=None):
        """Calculate the t=(self.order+1)^th order perturbation with the given ht

        Parameters
        ----------
        ht : Qobj
            Perturbation Hamiltonian of t^th order. It has a default value that is a zero operator.
        """
        # set default value of ht to zero operator
        if ht is None:
            ht = 0*self.hamiltonians[0]

        self.order += 1
        self.hamiltonians.append(ht)
        t = self.order

        # Do the calculation space-by-space to handle space split
        old_spaces = self.eigen_spaces
        self.eigen_spaces = []
        for sp in old_spaces:
            # calculate W using (1.2.2.3)
            dim = len(sp.base_set)
            W = zeros((dim, dim))
            energy_levels_sp = sp.energy_levels()
            for chi in range(dim):
                for xi in range(dim):
                    W[chi, xi] = (sp.base_set[chi].vectors[0].dag()*ht
                                  * sp.base_set[xi].vectors[0]).tr()
                    for i in range(1, t):
                        H = (sp.base_set[chi].vectors[0].dag()*self.hamiltonians[i]
                             * sp.base_set[xi].vectors[t-i]).tr()
                        E = energy_levels_sp[i].energy * (sp.base_set[chi].vectors[0].dag()
                             * sp.base_set[xi].vectors[t-i]).tr()
                        W[chi, xi] += H-E

            # solve the eigen value problem of W
            eigen_values, eigen_vectors = linalg.eig(W)
            # categorize eigenvalues and eigenvectors of W.
            eigen_system_info = self.categorize_eigenstates(eigen_values,[eigen_vectors[:, i] for i in range(dim)])
            # split each subspace
            self.eigen_spaces += sp.split(eigen_system_info)

        # calculate the state correction for t^th order
        for spn in self.eigen_spaces:  # each eigen space, n\xi in (1.2.3.1)
            energy_levels_n = spn.energy_levels()
            En = energy_levels_n[0]
            for base_xi in spn.base_set:
                vec_set_xi = base_xi.vectors
                # each base within the space, n\xi in (1.2.3.1)
                # calculate the t^th order correction of Phi_{n\xi}
                phit = vec_set_xi[0] * 0
                # any other way to get a zero state?
                for spm in self.eigen_spaces:
                    # each eigen space, m\alpha in (1.2.3.1)
                    energy_levels_m = spm.energy_levels()
                    Em = energy_levels_m[0]
                    if Em == En:
                        continue
                    for base_alpha in spm.base_set:
                        vec_set_alpha = base_alpha.vectors
                        # each base within the space, m\alpha in (1.2.3.1)
                        # calculate C
                        C = 0
                        for i in range(1, t+1):
                            H = (vec_set_alpha[0].dag()*self.hamiltonians[i]
                                 * vec_set_xi[t-i]).tr()
                            E = (energy_levels_n[i].energy
                                 * vec_set_alpha[0].dag()
                                 * vec_set_xi[t-i]).tr()
                            C += H - E
                        C /= (En.energy - Em.energy)
                        phit += C*vec_set_alpha[0]
                vec_set_xi += [phit]

    def goto_converge(self,lamda,tol=0):
        """Keep going to higher order perturbation corrections without perturbation in Hamiltonian
        until converge.
            
        Parameters
        ----------
        lamda: float
            The parameter lambda in the formula H = H0 + lambda*H1 + lambda^2*H2 + ...
        tol: float
            The tolerance used as criteria of converge. Default value is etol
        """
        if tol == 0:
            tol = self.etol

        converge = False
        while not converge:
            self.next_order()
            converge = True
            # get the newly calculated energies and states
            newe = []
            news = []
            for space in self.eigen_spaces:
                newe.append(space.elp_node.energy)
                for base in space.base_set:
                    news.append(base.vectors[-1])
            # check whether all energies converge
            for x in newe:
                if abs(x * lamda**self.order) > tol:
                    converge = False
                    break
            if not converge:
                continue
            # check whether all states converge
            for x in news:
                for i in range(x.dims[0][0]):
                    if abs(x[i,0] * lamda**self.order)>tol:
                        converge = False
                        break
                if not converge:
                    break
        print(self.order)

    def result(lamda):
        pass