from qutip import *
from numpy import zeros, linalg

# TODO: I kept using a lot of 'eigen', 'vectors', which might
# refer to different things in different situations clarify all
# the names to make them not confusing


class Perturbation:
    """This class is the calculator class that do the calculation
        of perturbation theory"""

    class EnergyLevel:
        """This class is the node of energy level tree"""

        def __init__(self, E, degeneracy):
            """Initialize a naked(not in any tree) node"""
            # Energy of this energy level
            self.E = E
            # degeneracy of this energy level
            self.degeneracy = degeneracy

    class EigenSpace:
        """This class is the entry of eigen_spaces"""
        def __init__(self, energy_levels=None, vectors=None):
            """Initialize an empty instance"""
            if energy_levels is None:
                energy_levels = []
            if vectors is None:
                vectors = []
            self.energy_levels = energy_levels
            self.vectors = vectors

        @classmethod
        def new0(cls, energy_level, vectors):
            """This function is usually used to initialize the parameter
                passed to the constructor of class Perturbation to create
                an instance for the eigen spaces of H0.

            Arguments:
            energy_level -- instance of class EnergyLevel for this energy level
            vectors -- list of eigen vectors for this energy level

            Return:
            An well-setted instance of class EigenSpace.
            """
            instance = cls([energy_level])
            for i in vectors:
                instance.vectors += [[i]]
            energy_level.eigen_space = instance
            return instance

        @classmethod
        def copy(cls, old_instance, new_energy_level, new_vectors):
            """This function is usually used to create an instance
                for the splitted space when the eigen space is splitted
                by the newly introduced perturbation.

            Arguments:

            old_instance -- the instance for the unsplitted space

            new_energy_level -- instance of class EnergyLevel for the
            new splitted energy level

            vectors -- list of new eigen vectors after splitted

            Return:
            An well-setted instance of class EigenSpace.
            """
            return cls(old_instance.energy_levels+[new_energy_level],
                       new_vectors)

    def __init__(self, H0, zero_order_energy_levels,
                 zero_order_eigen_spaces, Etol=1E-8):
        """Initialize the calculator class with H0 and its eigen values and vectors

        Argument:
        H0 -- the unperturbed Hamiltonian
        zero_order_energy_levels - list of class EnergyLevel instances
        corresponding to each energy level of H0

        zero_order_eigen_spaces -- list of class EigenSpace instances
        corresponding to the eigen space of each energy level

        Etol -- tolerance of energy, i.e. if |Em-En|<Etol, we regard
        the two energy are degenerate
        """
        # all the n^th order perturbation Hamiltonian
        self.Hamiltonians = [H0]
        # all the energy levels without perturbation,which is the roots
        # of energy level trees of the whole energy level forest
        self.energy_level_trees = zero_order_energy_levels
        # all the eigen spaces
        self.eigen_spaces = zero_order_eigen_spaces
        # the order of perturbation that we have calculated.
        self.order = 0
        # tolerance of energy
        self.Etol = Etol

    def next_order(self, Ht=None):
        """Calculate the t=(self.order+1)^th order perturbation with the given Ht

        Arguments:
        Ht -- Perturbation Hamiltonian of t^th order
        """
        # set default value of Ht to zero operator
        if Ht is None:
            Ht = 0*self.Hamiltonians[0]

        self.order += 1
        self.Hamiltonians.append(Ht)
        t = self.order

        # Do the calculation space-by-space to handle space split
        spaces_to_be_removed = []
        spaces_to_be_appended = []
        for sp in self.eigen_spaces:
            # calculate W using (1.2.2.3)
            dim = len(sp.vectors)
            W = zeros((dim, dim))
            for chi in range(dim):
                for xi in range(dim):
                    W[chi, xi] = (sp.vectors[chi][0].trans()*Ht
                                  * sp.vectors[xi][0]).tr()
                    for i in range(1, t):
                        H = (sp.vectors[chi][0].trans()*self.Hamiltonians[i]
                             * sp.vectors[xi][t-i]).tr()
                        E = (sp.energy_levels[i].E*sp.vectors[chi][0].trans()
                             * sp.vectors[xi][t-i]).tr()
                        W[chi, xi] += H-E

            # solve the eigen value problem of W
            eigen_values, eigen_vectors = linalg.eig(W)
            # classify eigen values and eigen vectors of W according
            # to degenerate. here we maintain a data structure:
            # [ [eigen value 1,[eigen vector 1, eigen vector 2, ...]],
            #   [eigen value 2,[eigen vector 1, eigen vector 2, ...]],
            #  ...  ]
            eigen_system_info = []
            for i in range(dim):
                eigen_value = eigen_values[i]
                eigen_vector = eigen_vectors[:, i]
                for j in eigen_system_info:
                    if abs(j[0] - eigen_value) < self.Etol:
                        # still degenerate
                        j[1] += [eigen_vector]
                        break
                else:
                    # not degenerate
                    eigen_system_info += [[eigen_value, [eigen_vector]]]

            # delete inappropriate "eigen_space" attribute in nodes of energy
            # level tree and remove old spaces(will be replaced by new and
            # splitted space) from self.eigen_spaces
            if len(eigen_system_info) > 1:
                # energy level splits
                for i in sp.energy_levels:
                    if hasattr(i, "eigen_space"):
                        del i.eigen_space
                spaces_to_be_removed += [sp]

            # create new entries in self.eigen_spaces to store the splitted
            # space and insert new nodes to energy level tree to store energy
            # of the t^th order perturbation
            new_nodes = []
            for i in eigen_system_info:
                E = i[0]
                eigen_vectors = i[1]
                degeneracy = len(eigen_vectors)
                # create new node in energy level tree
                node = self.EnergyLevel(E, degeneracy)
                node.prev_order = sp.energy_levels[t-1]
                # create new entry in self.eigen_spaces
                space = sp
                space.energy_levels += [node]
                if len(eigen_system_info) > 1:
                    # calculate new vectors
                    new_vectors = []
                    for j in eigen_vectors:
                        # each vector set of all energy level
                        new_vector_of_all_energy_level = []
                        for k in range(t):
                            # each order of perturbation within the same
                            # vector set
                            ket = j[0]*sp.vectors[0][k]
                            for l in range(1, dim):
                                # each element of eigen vector of W
                                ket += j[l] * sp.vectors[l][k]
                            new_vector_of_all_energy_level += [ket]
                        new_vectors += [new_vector_of_all_energy_level]
                    # create new EigenSpace entry
                    space = self.EigenSpace.copy(sp, node, new_vectors)
                    spaces_to_be_appended += [space]
                node.eigen_space = space
                new_nodes += [node]
            sp.energy_levels[t-1].next_order = new_nodes
        for i in spaces_to_be_removed:
            self.eigen_spaces.remove(i)
        self.eigen_spaces += spaces_to_be_appended

        # calculate the state correction for t^th order
        for spn in self.eigen_spaces:  # each eigen space, n\xi in (1.2.3.1)
            En = spn.energy_levels[0]
            for vec_set_xi in spn.vectors:
                # each base within the space, n\xi in (1.2.3.1)
                # calculate the t^th order correction of Phi_{n\xi}
                Phit = vec_set_xi[0] * 0
                # any other way to get a zero state?
                for spm in self.eigen_spaces:
                    # each eigen space, m\alpha in (1.2.3.1)
                    Em = spm.energy_levels[0]
                    if Em == En:
                        continue
                    for vec_set_alpha in spm.vectors:
                        # each base within the space, m\alpha in (1.2.3.1)
                        # calculate C
                        C = 0
                        for i in range(1, t+1):
                            H = (vec_set_alpha[0].trans()*self.Hamiltonians[i]
                                 * vec_set_xi[t-i]).tr()
                            E = (spn.energy_levels[i].E
                                 * vec_set_alpha[0].trans()
                                 * vec_set_xi[t-i]).tr()
                            C += H - E
                        C /= (En.E - Em.E)
                        Phit += C*vec_set_alpha[0]
                vec_set_xi += [Phit]
