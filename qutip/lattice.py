__all__ = ['Lattice1d', 'Lattice1d_f_Hubbard', 'Lattice1d_2c_hcb_Hubbard', 'Lattice1d_Heisenberg', 'cell_structures']

from scipy.sparse import (csr_matrix)
from qutip import (Qobj, tensor, basis, qeye, isherm, sigmax, sigmay, sigmaz)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


from scipy import sparse
from scipy.sparse.linalg import eigs
import operator as op
from functools import reduce
#from copy import deepcopy
import copy



def cell_structures(val_s=None, val_t=None, val_u=None):
    """
    Returns two matrices H_cell and cell_T to help the user form the inputs for
    defining an instance of Lattice1d and Lattice2d classes. The two matrices
    are the intra and inter cell Hamiltonians with the tensor structure of the
    specified site numbers and/or degrees of freedom defined by the user.

    Parameters
    ==========
    val_s : list of str/str
        The first list of str's specifying the sites/degrees of freedom in the
        unitcell

    val_t : list of str/str
        The second list of str's specifying the sites/degrees of freedom in the
        unitcell

    val_u : list of str/str
        The third list of str's specifying the sites/degrees of freedom in the
        unitcell

    Returns
    -------
    H_cell_s : list of list of str
        tensor structure of the cell Hamiltonian elements
    T_inter_cell_s : list of list of str
        tensor structure of the inter cell Hamiltonian elements
    H_cell : Qobj
        A Qobj initiated with all 0s with proper shape for an input as
        Hamiltonian_of_cell in Lattice1d.__init__()
    T_inter_cell : Qobj
        A Qobj initiated with all 0s with proper shape for an input as
        inter_hop in Lattice1d.__init__()
    """
    Er0_str = "At least one list of str necessary for using cell_structures!"
    Er1_str = "val_s is required to be a list of str's or a str"
    Er2_str = "val_t is required to be a list of str's or a str."
    Er3_str = "val_u is required to be a list of str's or a str"

    if val_s is None:
        raise Exception(Er0_str)
    if (not isinstance(val_s, list)) and (not isinstance(val_s, str)):
        raise Exception(Er1_str)
    if not all([isinstance(val, str) for val in val_s]):
        raise(Er1_str)

    if val_t is None:
        lng0 = len(val_s)
        row_i = np.arange(lng0).reshape(lng0, 1)
        row_i_c = np.ones(lng0).reshape(1, lng0)
        Rows = np.kron(row_i, row_i_c)
        Rows = np.array(Rows, dtype=int)

        H_cell_s = [[None for i in range(lng0)] for j in range(lng0)]
        T_inter_cell_s = [[None for i in range(lng0)] for j in range(lng0)]

        for ir in range(lng0):
            for ic in range(lng0):
                sst = val_s[Rows[ir][ic]]+" H "+val_s[Rows[ic][ir]]
                H_cell_s[ir][ic] = "<" + sst + ">"
                T_inter_cell_s[ir][ic] = "<cell(i):" + sst + ":cell(i+1) >"

    if val_t is not None and val_u is None:
        if not all([isinstance(val, str) for val in val_t]):
            raise Exception(Er2_str)
        lng0 = len(val_s)
        lng1 = len(val_t)
        p01 = lng0 * lng1
        srow_i = np.kron(np.arange(lng0), np.ones(lng1)).reshape(p01, 1)
        srow_i_c = np.ones(lng0 * lng1).reshape(1, p01)
        sRows = np.kron(srow_i, srow_i_c)
        sRows = np.array(sRows, dtype=int)

        trow_i = np.kron(np.ones(lng0), np.arange(lng1)).reshape(p01, 1)
        trow_i_c = np.ones(lng0 * lng1).reshape(1, p01)
        t_rows = np.kron(trow_i, trow_i_c)
        t_rows = np.array(t_rows, dtype=int)

        H_cell_s = [[None for i in range(p01)] for j in range(p01)]
        T_inter_cell_s = [[None for i in range(p01)] for j in range(p01)]

        for ir in range(p01):
            for jr in range(p01):
                sst = []
                sst.append(val_s[sRows[ir][jr]])
                sst.append(val_t[t_rows[ir][jr]])
                sst.append(" H ")
                sst.append(val_s[sRows[jr][ir]])
                sst.append(val_t[t_rows[jr][ir]])
                sst = ''.join(sst)
                H_cell_s[ir][jr] = "<" + sst + ">"

                llt = []
                llt.append("<cell(i):")
                llt.append(sst)
                llt.append(":cell(i+1) >")
                llt = ''.join(llt)
                T_inter_cell_s[ir][jr] = llt

    if val_u is not None:
        if not all([isinstance(val, str) for val in val_u]):
            raise Exception(Er3_str)
        lng0 = len(val_s)
        lng1 = len(val_t)
        lng2 = len(val_u)
        p012 = lng0 * lng1 * lng2
        srow_i = np.kron(np.arange(lng0), np.ones(lng1))
        srow_i = np.kron(srow_i, np.ones(lng2))
        srow_i = srow_i.reshape(p012, 1)
        srow_i_c = np.ones(p012).reshape(1, p012)
        sRows = np.kron(srow_i, srow_i_c)
        sRows = np.array(sRows, dtype=int)

        trow_i = np.kron(np.ones(lng0), np.arange(lng1))
        trow_i = np.kron(trow_i, np.ones(lng2))
        trow_i = trow_i.reshape(p012, 1)
        trow_i_c = np.ones(P).reshape(1, p012)
        t_rows = np.kron(trow_i, trow_i_c)
        t_rows = np.array(t_rows, dtype=int)

        urow_i = np.kron(np.ones(lng0), np.ones(lng1))
        urow_i = np.kron(urow_i, np.arange(lng2))
        urow_i = urow_i.reshape(p012, 1)
        urow_i_c = np.ones(p012).reshape(1, p012)
        uRows = np.kron(urow_i, urow_i_c)
        uRows = np.array(uRows, dtype=int)

        H_cell_s = [[None for i in range(p012)] for j in range(p012)]
        T_inter_cell_s = [[None for i in range(p012)] for j in range(p012)]

        for ir in range(p012):
            for jr in range(p012):
                sst = []
                sst.append(val_s[sRows[ir][jr]])
                sst.append(val_t[t_rows[ir][jr]])
                sst.append(val_u[uRows[ir][jr]])
                sst.append(" H ")
                sst.append(val_s[sRows[jr][ir]])
                sst.append(val_t[t_rows[jr][ir]])
                sst.append(val_u[uRows[jr][ir]])
                sst = ''.join(sst)
                H_cell_s[ir][jr] = "<" + sst + ">"

                llt = []
                llt.append("<cell(i):")
                llt.append(sst)
                llt.append(":cell(i+1) >")
                llt = ''.join(llt)
                T_inter_cell_s[ir][jr] = llt

    H_cell = np.zeros(np.shape(H_cell_s), dtype=complex)
    T_inter_cell = np.zeros(np.shape(T_inter_cell_s), dtype=complex)
    return (H_cell_s, T_inter_cell_s, H_cell, T_inter_cell)


class Lattice1d():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a
    specified number of unit cells in the crystal. It can return dispersion
    relationship, position operators, Hamiltonian in the position represention
    etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit
        cell.
    Hamiltonian_of_cell : qutip.Qobj
        The Hamiltonian of the unit cell.
    inter_hop : qutip.Qobj / list of Qobj
        The coupling between the unit cell at i and at (i+unit vector)

    Attributes
    ----------
    num_cell : int
        The number of unit cells in the crystal.
    cell_num_site : int
        The number of sites in a unit cell.
    length_for_site : int
        The length of the dimension per site of a unit cell.
    cell_tensor_config : list of int
        The tensor structure of the cell in the form
        [cell_num_site,cell_site_dof[:][0] ]
    lattice_tensor_config : list of int
        The tensor structure of the crystal in the
        form [num_cell,cell_num_site,cell_site_dof[:][0]]
    length_of_unit_cell : int
        The length of the dimension for a unit cell.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components
        along Cartesian unit vectors.
    H_intra : qutip.Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ qutip.Qobj
        The list of coupling terms between unit cells of the lattice.
    is_real : bool
        Indicates if the Hamiltonian is real or not.
    """
    def __init__(self, num_cell=10, boundary="periodic", cell_num_site=1,
                 cell_site_dof=[1], Hamiltonian_of_cell=None,
                 inter_hop=None):

        self.num_cell = num_cell
        self.cell_num_site = cell_num_site
        if (not isinstance(cell_num_site, int)) or cell_num_site < 0:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")

        if isinstance(cell_site_dof, list):
            l_v = 1
            for i, csd_i in enumerate(cell_site_dof):
                if (not isinstance(csd_i, int)) or csd_i < 0:
                    raise Exception("Invalid cell_site_dof list element at \
                                    index: ", i, "Elements of cell_site_dof \
                                    required to be positive integers.")
                l_v = l_v * cell_site_dof[i]
            self.cell_site_dof = cell_site_dof

        elif isinstance(cell_site_dof, int):
            if cell_site_dof < 0:
                raise Exception("cell_site_dof is required to be a positive \
                                integer.")
            else:
                l_v = cell_site_dof
                self.cell_site_dof = [cell_site_dof]
        else:
            raise Exception("cell_site_dof is required to be a positive \
                            integer or a list of positive integers.")
        self._length_for_site = l_v
        self.cell_tensor_config = [self.cell_num_site] + self.cell_site_dof
        self.lattice_tensor_config = [self.num_cell] + self.cell_tensor_config
        # remove any 1 present in self.cell_tensor_config and
        # self.lattice_tensor_config unless all the elements are 1

        if all(x == 1 for x in self.cell_tensor_config):
            self.cell_tensor_config = [1]
        else:
            while 1 in self.cell_tensor_config:
                self.cell_tensor_config.remove(1)

        if all(x == 1 for x in self.lattice_tensor_config):
            self.lattice_tensor_config = [1]
        else:
            while 1 in self.lattice_tensor_config:
                self.lattice_tensor_config.remove(1)

        dim_ih = [self.cell_tensor_config, self.cell_tensor_config]
        self._length_of_unit_cell = self.cell_num_site*self._length_for_site

        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

        if Hamiltonian_of_cell is None:       # There is no user input for
            # Hamiltonian_of_cell, so we set it ourselves
            H_site = np.diag(np.zeros(cell_num_site-1)-1, 1)
            H_site += np.diag(np.zeros(cell_num_site-1)-1, -1)
            if cell_site_dof == [1] or cell_site_dof == 1:
                Hamiltonian_of_cell = Qobj(H_site, type='oper')
                self._H_intra = Hamiltonian_of_cell
            else:
                Hamiltonian_of_cell = tensor(Qobj(H_site),
                                             qeye(self.cell_site_dof))
                dih = Hamiltonian_of_cell.dims[0]
                if all(x == 1 for x in dih):
                    dih = [1]
                else:
                    while 1 in dih:
                        dih.remove(1)
                self._H_intra = Qobj(Hamiltonian_of_cell, dims=[dih, dih],
                                     type='oper')
        elif not isinstance(Hamiltonian_of_cell, Qobj):    # The user
            # input for Hamiltonian_of_cell is not a Qobj and hence is invalid
            raise Exception("Hamiltonian_of_cell is required to be a Qobj.")
        else:       # We check if the user input Hamiltonian_of_cell have the
            # right shape or not. If approved, we give it the proper dims
            # ourselves.
            r_shape = (self._length_of_unit_cell, self._length_of_unit_cell)
            if Hamiltonian_of_cell.shape != r_shape:
                raise Exception("Hamiltonian_of_cell does not have a shape \
                            consistent with cell_num_site and cell_site_dof.")
            self._H_intra = Qobj(Hamiltonian_of_cell, dims=dim_ih, type='oper')
        is_real = np.isreal(self._H_intra.full()).all()
        if not isherm(self._H_intra):
            raise Exception("Hamiltonian_of_cell is required to be Hermitian.")

        nSb = self._H_intra.shape
        if isinstance(inter_hop, list):      # There is a user input list
            inter_hop_sum = Qobj(np.zeros(nSb))
            for i in range(len(inter_hop)):
                if not isinstance(inter_hop[i], Qobj):
                    raise Exception("inter_hop[", i, "] is not a Qobj. All \
                                inter_hop list elements need to be Qobj's. \n")
                nSi = inter_hop[i].shape
                # inter_hop[i] is a Qobj, now confirmed
                if nSb != nSi:
                    raise Exception("inter_hop[", i, "] is dimensionally \
                        incorrect. All inter_hop list elements need to \
                        have the same dimensionality as Hamiltonian_of_cell.")
                else:    # inter_hop[i] has the right shape, now confirmed,
                    inter_hop[i] = Qobj(inter_hop[i], dims=dim_ih)
                    inter_hop_sum = inter_hop_sum + inter_hop[i]
                    is_real = is_real and np.isreal(inter_hop[i].full()).all()
            self._H_inter_list = inter_hop    # The user input list was correct
            # we store it in _H_inter_list
            self._H_inter = Qobj(inter_hop_sum, dims=dim_ih, type='oper')
        elif isinstance(inter_hop, Qobj):  # There is a user input
            # Qobj
            nSi = inter_hop.shape
            if nSb != nSi:
                raise Exception("inter_hop is required to have the same \
                dimensionality as Hamiltonian_of_cell.")
            else:
                inter_hop = Qobj(inter_hop, dims=dim_ih, type='oper')
            self._H_inter_list = [inter_hop]
            self._H_inter = inter_hop
            is_real = is_real and np.isreal(inter_hop.full()).all()

        elif inter_hop is None:      # inter_hop is the default None)
            # So, we set self._H_inter_list from cell_num_site and
            # cell_site_dof
            if self._length_of_unit_cell == 1:
                inter_hop = Qobj([[-1]], type='oper')
            else:
                bNm = basis(cell_num_site, cell_num_site-1)
                bN0 = basis(cell_num_site, 0)
                siteT = -bNm * bN0.dag()
                inter_hop = tensor(Qobj(siteT), qeye(self.cell_site_dof))
            dih = inter_hop.dims[0]
            if all(x == 1 for x in dih):
                dih = [1]
            else:
                while 1 in dih:
                    dih.remove(1)
            self._H_inter_list = [Qobj(inter_hop, dims=[dih, dih],
                                       type='oper')]
            self._H_inter = Qobj(inter_hop, dims=[dih, dih], type='oper')
        else:
            raise Exception("inter_hop is required to be a Qobj or a \
                            list of Qobjs.")

        self.positions_of_sites = [(i/self.cell_num_site) for i in
                                   range(self.cell_num_site)]
        self._inter_vec_list = [[1] for i in range(len(self._H_inter_list))]
        self._Brav_lattice_vectors_list = [[1]]     # unit vectors
        self.is_real = is_real

    def __repr__(self):
        s = ""
        s += ("Lattice1d object: " +
              "Number of cells = " + str(self.num_cell) +
              ",\nNumber of sites in the cell = " + str(self.cell_num_site) +
              ",\nDegrees of freedom per site = " +
              str(
               self.lattice_tensor_config[2:len(self.lattice_tensor_config)]) +
              ",\nLattice tensor configuration = " +
              str(self.lattice_tensor_config) +
              ",\nbasis_Hamiltonian = " + str(self._H_intra) +
              ",\ninter_hop = " + str(self._H_inter_list) +
              ",\ncell_tensor_config = " + str(self.cell_tensor_config) +
              "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def Hamiltonian(self):
        """
        Returns the lattice Hamiltonian for the instance of Lattice1d.

        Returns
        ----------
        Qobj(Hamil) : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        """
        D = qeye(self.num_cell)
        T = np.diag(np.zeros(self.num_cell-1)+1, 1)
        Tdag = np.diag(np.zeros(self.num_cell-1)+1, -1)

        if self.period_bnd_cond_x == 1 and self.num_cell > 2:
            Tdag[0][self.num_cell-1] = 1
            T[self.num_cell-1][0] = 1
        T = Qobj(T)
        Tdag = Qobj(Tdag)
        Hamil = tensor(D, self._H_intra) + tensor(
                T, self._H_inter) + tensor(Tdag, self._H_inter.dag())
        dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(Hamil, dims=dim_H)

    def basis(self, cell, site, dof_ind):
        """
        Returns a single particle wavefunction ket with the particle localized
        at a specified dof at a specified site of a specified cell.

        Parameters
        -------
        cell : int
            The cell at which the particle is to be localized.

        site : int
            The site of the cell at which the particle is to be localized.

        dof_ind : int/ list of int
            The index of the degrees of freedom with which the sigle particle
            is to be localized.

        Returns
        ----------
        vec_i : qutip.Qobj
            ket type Quantum object representing the localized particle.
        """
        if not isinstance(cell, int):
            raise Exception("cell needs to be int in basis().")
        elif cell >= self.num_cell:
            raise Exception("cell needs to less than Lattice1d.num_cell")

        if not isinstance(site, int):
            raise Exception("site needs to be int in basis().")
        elif site >= self.cell_num_site:
            raise Exception("site needs to less than Lattice1d.cell_num_site.")

        if isinstance(dof_ind, int):
            dof_ind = [dof_ind]

        if not isinstance(dof_ind, list):
            raise Exception("dof_ind in basis() needs to be an int or \
                            list of int")

        if np.shape(dof_ind) == np.shape(self.cell_site_dof):
            for i in range(len(dof_ind)):
                if dof_ind[i] >= self.cell_site_dof[i]:
                    raise Exception("in basis(), dof_ind[", i, "] is required\
                                to be smaller than cell_num_site[", i, "]")
        else:
            raise Exception("dof_ind in basis() needs to be of the same \
                            dimensions as cell_site_dof.")

        doft = basis(self.cell_site_dof[0], dof_ind[0])
        for i in range(1, len(dof_ind)):
            doft = tensor(doft, basis(self.cell_site_dof[i], dof_ind[i]))
        vec_i = tensor(
                basis(self.num_cell, cell), basis(self.cell_num_site, site),
                doft)
        ltc = self.lattice_tensor_config
        vec_i = Qobj(vec_i, dims=[ltc, [1 for i, j in enumerate(ltc)]])
        return vec_i

    def distribute_operator(self, op):
        """
        A function that returns an operator matrix that applies op to all the
        cells in the 1d lattice

        Parameters
        -------
        op : qutip.Qobj
            Qobj representing the operator to be applied at all cells.

        Returns
        ----------
        op_H : qutip.Qobj
            Quantum object representing the operator with op applied at all
            cells.
        """
        nSb = self._H_intra.shape
        if not isinstance(op, Qobj):
            raise Exception("op in distribute_operator() needs to be Qobj.\n")
        nSi = op.shape
        if nSb != nSi:
            raise Exception("op in distribute_operstor() is required to \
            have the same dimensionality as Hamiltonian_of_cell.")
        cell_All = list(range(self.num_cell))
        op_H = self.operator_at_cells(op, cells=cell_All)
        return op_H

    def x(self):
        """
        Returns the position operator. All degrees of freedom has the cell
        number at their correspondig entry in the position operator.

        Returns
        -------
        Qobj(xs) : qutip.Qobj
            The position operator.
        """
        nx = self.cell_num_site
        ne = self._length_for_site
#        positions = np.kron(range(nx), [1/nx for i in range(ne)])  # not used
        # in the current definition of x
#        S = np.kron(np.ones(self.num_cell), positions)
#        xs = np.diagflat(R+S)        # not used in the
        # current definition of x
        R = np.kron(range(0, self.num_cell), np.ones(nx*ne))
        xs = np.diagflat(R)
        dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(xs, dims=dim_H)

    def k(self):
        """
        Returns the crystal momentum operator. All degrees of freedom has the
        cell number at their correspondig entry in the position operator.

        Returns
        -------
        Qobj(ks) : qutip.Qobj
            The crystal momentum operator in units of 1/a. L is the number
            of unit cells, a is the length of a unit cell which is always taken
            to be 1.
        """
        L = self.num_cell
        kop = np.zeros((L, L), dtype=complex)
        for row in range(L):
            for col in range(L):
                if row == col:
                    kop[row, col] = (L-1)/2
#                    kop[row, col] = ((L+1) % 2)/ 2
                    # shifting the eigenvalues
                else:
                    kop[row, col] = 1/(np.exp(2j * np.pi * (row - col)/L) - 1)
        qkop = Qobj(kop)
        [kD, kV] = qkop.eigenstates()
        kop_P = np.zeros((L, L), dtype=complex)
        for eno in range(L):
            if kD[eno] > (L // 2 + 0.5):
                vl = kD[eno] - L
            else:
                vl = kD[eno]
            vk = kV[eno]
            kop_P = kop_P + vl * vk * vk.dag()
        kop = 2 * np.pi / L * kop_P
        nx = self.cell_num_site
        ne = self._length_for_site
        k = np.kron(kop, np.eye(nx*ne))
        dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(k, dims=dim_H)

    def operator_at_cells(self, op, cells):
        """
        A function that returns an operator matrix that applies op to specific
        cells specified in the cells list

        Parameters
        ----------
        op : qutip.Qobj
            Qobj representing the operator to be applied at certain cells.

        cells: list of int
            The cells at which the operator op is to be applied.

        Returns
        -------
        Qobj(op_H) : Qobj
            Quantum object representing the operator with op applied at
            the specified cells.
        """
        if isinstance(cells, int):
            cells = [cells]
        if isinstance(cells, list):
            for i, cells_i in enumerate(cells):
                if not isinstance(cells_i, int):
                    raise Exception("cells[", i, "] is not an int!elements of \
                                    cells is required to be ints.")
        else:
            raise Exception("cells in operator_at_cells() need to be an int or\
                               a list of ints.")

        nSb = self._H_intra.shape
        if (not isinstance(op, Qobj)):
            raise Exception("op in operator_at_cells need to be Qobj's. \n")
        nSi = op.shape
        if (nSb != nSi):
            raise Exception("op in operstor_at_cells() is required to \
                            be dimensionaly the same as Hamiltonian_of_cell.")

        (xx, yy) = op.shape
        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])
        nS = self._length_of_unit_cell
        nx_units = self.num_cell
        ny_units = 1
        for i in range(nx_units):
            lin_RI = i
            if (i in cells):
                for k in range(xx):
                    for l in range(yy):
                        row_ind = np.append(row_ind, [lin_RI*nS+k])
                        col_ind = np.append(col_ind, [lin_RI*nS+l])
                        data = np.append(data, [op[k, l]])

        m = nx_units*ny_units*nS
        op_H = csr_matrix((data, (row_ind, col_ind)), [m, m],
                          dtype=np.complex128)
        dim_op = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(op_H, dims=dim_op)

    def operator_between_cells(self, op, row_cell, col_cell):
        """
        A function that returns an operator matrix that applies op to specific
        cells specified in the cells list

        Parameters
        ----------
        op : qutip.Qobj
            Qobj representing the operator to be put between cells row_cell
            and col_cell.

        row_cell: int
            The row index for cell for the operator op to be applied.

        col_cell: int
            The column index for cell for the operator op to be applied.

        Returns
        -------
        oper_bet_cell : Qobj
            Quantum object representing the operator with op applied between
            the specified cells.
        """
        if not isinstance(row_cell, int):
            raise Exception("row_cell is required to be an int between 0 and\
                            num_cell - 1.")
            if row_cell < 0 or row_cell > self.num_cell-1:
                raise Exception("row_cell is required to be an int between 0\
                                and num_cell - 1.")
        if not isinstance(col_cell, int):
            raise Exception("row_cell is required to be an int between 0 and\
                            num_cell - 1.")
            if col_cell < 0 or col_cell > self.num_cell-1:
                raise Exception("row_cell is required to be an int between 0\
                                and num_cell - 1.")

        nSb = self._H_intra.shape
        if (not isinstance(op, Qobj)):
            raise Exception("op in operator_between_cells need to be Qobj's.")
        nSi = op.shape
        if (nSb != nSi):
            raise Exception("op in operstor_between_cells() is required to \
                            be dimensionally the same as Hamiltonian_of_cell.")

        T = np.zeros((self.num_cell, self.num_cell), dtype=complex)
        T[row_cell, col_cell] = 1
        op_H = np.kron(T, op)
        dim_op = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(op_H, dims=dim_op)

    def plot_dispersion(self):
        """
        Plots the dispersion relationship for the lattice with the specified
        number of unit cells. The dispersion of the infinte crystal is also
        plotted if num_cell is smaller than MAXc.
        """
        MAXc = 20     # Cell numbers above which we do not plot the infinite
        # crystal dispersion
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")

        if self.num_cell <= MAXc:
            (kxA, val_ks) = self.get_dispersion(101)
        (knxA, val_kns) = self.get_dispersion()
        fig, ax = plt.subplots()
        if self.num_cell <= MAXc:
            for g in range(self._length_of_unit_cell):
                ax.plot(kxA/np.pi, val_ks[g, :])

        for g in range(self._length_of_unit_cell):
            if self.num_cell % 2 == 0:
                ax.plot(np.append(knxA, [np.pi])/np.pi,
                        np.append(val_kns[g, :], val_kns[g, 0]), 'ro')
            else:
                ax.plot(knxA/np.pi, val_kns[g, :], 'ro')
        ax.set_ylabel('Energy')
        ax.set_xlabel(r'$k_x(\pi/a)$')
        plt.show(fig)
        fig.savefig('./Dispersion.pdf')

    def get_dispersion(self, knpoints=0):
        """
        Returns dispersion relationship for the lattice with the specified
        number of unit cells with a k array and a band energy array.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        val_kns : np.array
            val_kns[j][:] is the array of band energies of the jth band good at
            all the good Quantum numbers of k.
        """
        # The _k_space_calculations() function is not used for get_dispersion
        # because we calculate the infinite crystal dispersion in
        # plot_dispersion using this coode and we do not want to calculate
        # all the eigen-values, eigenvectors of the bulk Hamiltonian for too
        # many points, as is done in the _k_space_calculations() function.
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        if knpoints == 0:
            knpoints = self.num_cell

        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        val_kns = np.zeros((self._length_of_unit_cell, knpoints), dtype=float)
        knxA = np.zeros((knpoints, 1), dtype=float)
        G0_H = self._H_intra
#        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))

        for ks in range(knpoints):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)

            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))

        for ks in range(knpoints):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()
            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k)
            (vals, veks) = qH_k.eigenstates()
            val_kns[:, ks] = vals[:]
        return (knxA, val_kns)

    def bloch_wave_functions(self):
        r"""
        Returns eigenvectors ($\psi_n(k)$) of the Hamiltonian in a
        numpy.ndarray for translationally symmetric lattices with periodic
        boundary condition.

        .. math::
            :nowrap:

            \begin{eqnarray}
            |\psi_n(k) \rangle = |k \rangle \otimes | u_{n}(k) \rangle   \\
            | u_{n}(k) \rangle = a_n(k)|a\rangle  + b_n(k)|b\rangle \\
            \end{eqnarray}

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        eigenstates : ordered np.array
            eigenstates[j][0] is the jth eigenvalue.
            eigenstates[j][1] is the corresponding eigenvector.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        dtype = [('eigen_value', np.longdouble), ('eigen_vector', Qobj)]
        values = list()
        for i in range(self.num_cell):
            for j in range(self._length_of_unit_cell):
                values.append((
                        val_kns[j][i], vec_xs[j+i*self._length_of_unit_cell]))
        eigen_states = np.array(values, dtype=dtype)
#        eigen_states = np.sort(eigen_states, order='eigen_value')
        return eigen_states

    def cell_periodic_parts(self):
        r"""
        Returns eigenvectors of the bulk Hamiltonian, i.e. the cell periodic
        part($u_n(k)$) of the Bloch wavefunctios in a numpy.ndarray for
        translationally symmetric lattices with periodic boundary condition.

        .. math::
            :nowrap:

            \begin{eqnarray}
            |\psi_n(k) \rangle = |k \rangle \otimes | u_{n}(k) \rangle   \\
            | u_{n}(k) \rangle = a_n(k)|a\rangle  + b_n(k)|b\rangle \\
            \end{eqnarray}

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        vec_kns : np.ndarray of Qobj's
            vec_kns[j] is the Oobj of type ket that holds an eigenvector of the
            bulk Hamiltonian of the lattice.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        return (knxA, vec_kns)

    def bulk_Hamiltonians(self):
        """
        Returns the bulk momentum space Hamiltonian ($H(k)$) for the lattice at
        the good quantum numbers of k in a numpy ndarray of Qobj's.

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        qH_ks : np.ndarray of Qobj's
            qH_ks[j] is the Oobj of type oper that holds a bulk Hamiltonian
            for a good quantum number k.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        return (knxA, qH_ks)

    def _k_space_calculations(self, knpoints=0):
        """
        Returns bulk Hamiltonian, its eigenvectors and eigenvectors of the
        space Hamiltonian at all the good quantum numbers of a periodic
        translationally invariant lattice.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        qH_ks : np.ndarray of Qobj's
            qH_ks[j] is the Oobj of type oper that holds a bulk Hamiltonian
            for a good quantum number k.

        vec_xs : np.ndarray of Qobj's
            vec_xs[j] is the Oobj of type ket that holds an eigenvector of the
            Hamiltonian of the lattice.

        vec_kns : np.ndarray of Qobj's
            vec_kns[j] is the Oobj of type ket that holds an eigenvector of the
            bulk Hamiltonian of the lattice.
        """
        if knpoints == 0:
            knpoints = self.num_cell

        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        val_kns = np.zeros((self._length_of_unit_cell, knpoints), dtype=float)
        knxA = np.zeros((knpoints, 1), dtype=float)
        G0_H = self._H_intra
        vec_kns = np.zeros((knpoints, self._length_of_unit_cell,
                           self._length_of_unit_cell), dtype=complex)

        vec_xs = np.array([None for i in range(
                knpoints * self._length_of_unit_cell)])
        qH_ks = np.array([None for i in range(knpoints)])

        for ks in range(knpoints):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)

            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))
        dim_hk = [self.cell_tensor_config, self.cell_tensor_config]
        for ks in range(knpoints):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()
            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k, dims=dim_hk)
            qH_ks[ks] = qH_k
            (vals, veks) = qH_k.eigenstates()
            plane_waves = np.kron(np.exp(-1j * (kx * range(self.num_cell))),
                                  np.ones(self._length_of_unit_cell))

            for eig_no in range(self._length_of_unit_cell):
                unit_cell_periodic = np.kron(
                    np.ones(self.num_cell), veks[eig_no].dag())
                vec_x = np.multiply(plane_waves, unit_cell_periodic)

                dim_H = [list(np.ones(len(self.lattice_tensor_config),
                                      dtype=int)), self.lattice_tensor_config]
                if self.is_real:
                    if np.count_nonzero(vec_x) > 0:
                        vec_x = np.real(vec_x)

                length_vec_x = np.sqrt((Qobj(vec_x) * Qobj(vec_x).dag())[0][0])
                vec_x = vec_x / length_vec_x
                vec_x = Qobj(vec_x, dims=dim_H)
                vec_xs[ks*self._length_of_unit_cell+eig_no] = vec_x.dag()

            for i in range(self._length_of_unit_cell):
                v0 = np.squeeze(veks[i].full(), axis=1)
                vec_kns[ks, i, :] = v0
            val_kns[:, ks] = vals[:]

        return (knxA, qH_ks, val_kns, vec_kns, vec_xs)

    def winding_number(self):
        """
        Returns the winding number for a lattice that has chiral symmetry and
        also plots the trajectory of (dx,dy)(dx,dy are the coefficients of
        sigmax and sigmay in the Hamiltonian respectively) on a plane.

        Returns
        -------
        winding_number : int or str
            knxA[j][0] is the jth good Quantum number k.
        """
        winding_number = 'defined'
        if (self._length_of_unit_cell != 2):
            raise Exception('H(k) is not a 2by2 matrix.')

        if (self._H_intra[0, 0] != 0 or self._H_intra[1, 1] != 0):
            raise Exception("Hamiltonian_of_cell has nonzero diagonals!")

        for i in range(len(self._H_inter_list)):
            H_I_00 = self._H_inter_list[i][0, 0]
            H_I_11 = self._H_inter_list[i][1, 1]
            if (H_I_00 != 0 or H_I_11 != 0):
                raise Exception("inter_hop has nonzero diagonal elements!")

        chiral_op = self.distribute_operator(sigmaz())
        Hamt = self.Hamiltonian()
        anti_commutator_chi_H = chiral_op * Hamt + Hamt * chiral_op
        is_null = (np.abs(anti_commutator_chi_H.full()) < 1E-10).all()

        if not is_null:
            raise Exception("The Hamiltonian does not have chiral symmetry!")

        knpoints = 100  # choose even
        kn_start = 0
        kn_end = 2*np.pi

        knxA = np.zeros((knpoints+1, 1), dtype=float)
        G0_H = self._H_intra
        mx_k = np.array([None for i in range(knpoints+1)])
        my_k = np.array([None for i in range(knpoints+1)])
        Phi_m_k = np.array([None for i in range(knpoints+1)])

        for ks in range(knpoints+1):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)
            knxA[ks, 0] = knx

        for ks in range(knpoints+1):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()
            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k)
            mx_k[ks] = 0.5*(qH_k*sigmax()).tr()
            my_k[ks] = 0.5*(qH_k*sigmay()).tr()

            if np.abs(mx_k[ks]) < 1E-10 and np.abs(my_k[ks]) < 1E-10:
                winding_number = 'undefined'

            if np.angle(mx_k[ks]+1j*my_k[ks]) >= 0:
                Phi_m_k[ks] = np.angle(mx_k[ks]+1j*my_k[ks])
            else:
                Phi_m_k[ks] = 2*np.pi + np.angle(mx_k[ks]+1j*my_k[ks])

        if winding_number == 'defined':
            ddk_Phi_m_k = np.roll(Phi_m_k, -1) - Phi_m_k
            intg_over_k = -np.sum(ddk_Phi_m_k[0:knpoints//2])+np.sum(
                    ddk_Phi_m_k[knpoints//2:knpoints])
            winding_number = intg_over_k/(2*np.pi)

            X_lim = 1.125 * np.abs(self._H_intra.full()[1, 0])
            for i in range(len(self._H_inter_list)):
                X_lim = X_lim + np.abs(self._H_inter_list[i].full()[1, 0])
            plt.figure(figsize=(3*X_lim, 3*X_lim))
            plt.plot(mx_k, my_k)
            plt.plot(0, 0, 'ro')
            plt.ylabel('$h_y$')
            plt.xlabel('$h_x$')
            plt.xlim(-X_lim, X_lim)
            plt.ylim(-X_lim, X_lim)
            plt.grid()
            plt.show()
            plt.savefig('./Winding.pdf')
            plt.close()
        return winding_number

    def _unit_site_H(self):
        """
        Returns a site's Hamiltonian part.

        Returns
        -------
        Hcell : list of Qobj's'
            Hcell[i][j] is the site's Hamiltonian part.
        """
        CNS = self.cell_num_site
        Hcell = [[{} for i in range(CNS)] for j in range(CNS)]

        for i0 in range(CNS):
            for j0 in range(CNS):
                Qin = np.zeros((self._length_for_site, self._length_for_site),
                               dtype=complex)
                for i in range(self._length_for_site):
                    for j in range(self._length_for_site):
                        Qin[i, j] = self._H_intra[
                                i0*self._length_for_site+i,
                                j0*self._length_for_site+j]
                if len(self.cell_tensor_config) > 1:
                    dim_site = list(filter(lambda a: a != 1,
                                           self.cell_tensor_config))
                dim_site = self.cell_tensor_config
                dims_site = [dim_site, dim_site]
                Hcell[i0][j0] = Qobj(Qin, dims=dims_site)

        return Hcell

    def display_unit_cell(self, label_on=False):
        """
        Produces a graphic displaying the unit cell features with labels on if
        defined by user. Also returns a dict of Qobj's corresponding to the
        labeled elements on the display.

        Returns
        -------
        Hcell : dict
            Hcell[i][j] is the Hamiltonian segment for $H_{i,j}$ labeled on the
            graphic.
        """
        CNS = self.cell_num_site
        Hcell = self._unit_site_H()

        fig = plt.figure(figsize=[CNS*2, CNS*2.5])
        ax = fig.add_subplot(111, aspect='equal')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        i = self._length_for_site
        if (CNS == 1):
            ax.plot([self.positions_of_sites[0]], [0], "o", c="b", mec="w",
                    mew=0.0, zorder=10, ms=8.0)
            if label_on is True:
                plt.text(x=self.positions_of_sites[0]+0.2, y=0.0,
                         s='H'+str(i)+str(i), horizontalalignment='center',
                         verticalalignment='center')
            x2 = (1+self.positions_of_sites[CNS-1])/2
            x1 = x2-1
            h = 1-x2
            ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7)
            plt.axis('off')
            plt.show()
            plt.close()
        else:

            for i in range(CNS):
                ax.plot([self.positions_of_sites[i]], [0], "o", c="b", mec="w",
                        mew=0.0, zorder=10, ms=8.0)
                if label_on is True:
                    x_b = self.positions_of_sites[i]+1/CNS/6
                    plt.text(x=x_b, y=0.0, s='H'+str(i)+str(i),
                             horizontalalignment='center',
                             verticalalignment='center')
                if i == CNS-1:
                    continue

                for j in range(i+1, CNS):
                    if (Hcell[i][j].full() == 0).all():
                        continue
                    c_cen = (self.positions_of_sites[
                            i]+self.positions_of_sites[j])/2

                    c_radius = (self.positions_of_sites[
                            j]-self.positions_of_sites[i])/2

                    circle1 = plt.Circle((c_cen, 0), c_radius, color='g',
                                         fill=False)
                    ax.add_artist(circle1)
                    if label_on is True:
                        x_b = c_cen
                        y_b = c_radius - 0.025
                        plt.text(x=x_b, y=y_b, s='H'+str(i)+str(j),
                                 horizontalalignment='center',
                                 verticalalignment='center')
            x2 = (1+self.positions_of_sites[CNS-1])/2
            x1 = x2-1
            h = (self.positions_of_sites[
                    CNS-1]-self.positions_of_sites[0])*8/15
            ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7)
            plt.axis('off')
            plt.show()
            plt.close()
        return Hcell

    def display_lattice(self):
        """
        Produces a graphic portraying the lattice symbolically with a unit cell
        marked in it.

        Returns
        -------
        inter_T : Qobj
            The coefficient of $\psi_{i,N}^{\dagger}\psi_{0,i+1}$, i.e. the
            coupling between the two boundary sites of the two unit cells i and
            i+1.
        """
        Hcell = self._unit_site_H()
        dims_site = Hcell[0][0].dims

        dim_I = [self.cell_tensor_config, self.cell_tensor_config]
        csn = self.cell_num_site
        H_inter = Qobj(np.zeros((self._length_of_unit_cell,
                                 self._length_of_unit_cell)), dims=dim_I)
        for no, inter_hop_no in enumerate(self._H_inter_list):
            H_inter = H_inter + inter_hop_no

        H_inter = np.array(H_inter)

        j0 = 0
        i0 = csn-1
        Qin = np.zeros((self._length_for_site, self._length_for_site),
                       dtype=complex)
        for i in range(self._length_for_site):
            for j in range(self._length_for_site):
                Qin[i, j] = H_inter[i0*self._length_for_site+i,
                                    j0*self._length_for_site+j]
        inter_T = Qin

        fig = plt.figure(figsize=[self.num_cell*3, self.num_cell*3])
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = fig.add_subplot(111, aspect='equal')

        for nc in range(self.num_cell):
            x_cell = nc
            for i in range(csn):
                ax.plot([x_cell + self.positions_of_sites[i]], [0], "o",
                        c="b", mec="w", mew=0.0, zorder=10, ms=8.0)

                if nc > 0:
                    # plot inter_cell_hop
                    ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                             x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                            "-", c="r", lw=1.5, zorder=7)

                    x_b = (x_cell-1+self.positions_of_sites[
                            csn-1] + x_cell + self.positions_of_sites[0])/2

                    plt.text(x=x_b, y=0.1, s='T',
                             horizontalalignment='center',
                             verticalalignment='center')
                if i == csn-1:
                    continue

                for j in range(i+1, csn):

                    if (Hcell[i][j].full() == 0).all():
                        continue
                    c_cen = self.positions_of_sites[i]
                    c_cen = (c_cen+self.positions_of_sites[j])/2
                    c_cen = c_cen + x_cell

                    c_radius = self.positions_of_sites[j]
                    c_radius = (c_radius-self.positions_of_sites[i])/2

                    circle1 = plt.Circle((c_cen, 0),
                                         c_radius, color='g', fill=False)
                    ax.add_artist(circle1)
        if (self.period_bnd_cond_x == 1):
            x_cell = 0
            x_b = 2*x_cell-1+self.positions_of_sites[csn-1]
            x_b = (x_b+self.positions_of_sites[0])/2

            plt.text(x=x_b, y=0.1, s='T', horizontalalignment='center',
                     verticalalignment='center')
            ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                     x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                    "-", c="r", lw=1.5, zorder=7)

            x_cell = self.num_cell
            x_b = 2*x_cell-1+self.positions_of_sites[csn-1]
            x_b = (x_b+self.positions_of_sites[0])/2

            plt.text(x=x_b, y=0.1, s='T', horizontalalignment='center',
                     verticalalignment='center')
            ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                     x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                    "-", c="r", lw=1.5, zorder=7)

        x2 = (1+self.positions_of_sites[csn-1])/2
        x1 = x2-1
        h = 0.5

        if self.num_cell > 2:
            xu = 1    # The index of cell over which the black box is drawn
            x1 = x1+xu
            x2 = x2+xu
        ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        plt.axis('off')
        plt.show()
        plt.close()

        return Qobj(inter_T, dims=dims_site)

class Lattice1d_f_Hubbard():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a
    specified number of unit cells in the crystal. It can return dispersion
    relationship, position operators, Hamiltonian in the position represention
    etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit
        cell.
    Hamiltonian_of_cell : qutip.Qobj
        The Hamiltonian of the unit cell.
    inter_hop : qutip.Qobj / list of Qobj
        The coupling between the unit cell at i and at (i+unit vector)

    Attributes
    ----------
    num_cell : int
        The number of unit cells in the crystal.
    cell_num_site : int
        The number of sites in a unit cell.
    length_for_site : int
        The length of the dimension per site of a unit cell.
    cell_tensor_config : list of int
        The tensor structure of the cell in the form
        [cell_num_site,cell_site_dof[:][0] ]
    lattice_tensor_config : list of int
        The tensor structure of the crystal in the
        form [num_cell,cell_num_site,cell_site_dof[:][0]]
    length_of_unit_cell : int
        The length of the dimension for a unit cell.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components
        along Cartesian unit vectors.
    H_intra : qutip.Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ qutip.Qobj
        The list of coupling terms between unit cells of the lattice.
    is_real : bool
        Indicates if the Hamiltonian is real or not.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1,
                 U=1, fillingUp=None, fillingDown=None, k=None):

        self.paramT = t
        self.paramU = U
        self.latticeSize = [num_sites]
        self.fillingUp = fillingUp
        self.fillingDown = fillingDown
        self.kval = k

        if (not isinstance(num_sites, int)) or num_sites > 18:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")
                            
        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_Hubbard object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n on-site interaction energy, U = " +
              str(self.U ) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def basis_fermionic_Hubbard_chain(self):
        """
        Produces a graphic portraying the lattice symbolically with a unit cell
        marked in it.

        Returns
        -------
        inter_T : Qobj
            The coefficient of $\psi_{i,N}^{\dagger}\psi_{0,i+1}$, i.e. the
            coupling between the two boundary sites of the two unit cells i and
            i+1.
        """
        latticeType = 'cubic'
        [indNeighbors, nSites] = self._getNearestNeighbors(latticeType=latticeType, latticeSize=self.latticeSize)

        nStatesUp = self._ncr(nSites, self.fillingUp)
        nStatesDown = self._ncr(nSites, self.fillingDown)
        linearLatticeSizeY = self.latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/linearLatticeSizeY)
        [basisStatesUp, intStatesUp, indOnesUp] = self.createHeisenbergBasis(nStatesUp, nSites, self.fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown] = self.createHeisenbergBasis(nStatesDown, nSites, self.fillingDown)
        #print( np.shape(basisStatesUp) )
        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp] = self._findReprOnlyTrans(basisStatesUp, self.latticeSize)
        #print("Freshly baked:: symOpInvariantsUp:  ", symOpInvariantsUp)
        #print("Freshly baked:: index2ReprUp:  ", index2ReprUp)
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStatesDown*bin2dez, axis=1)
        intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        kValue = kVector[self.kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates] = self._combine2HubbardBasisOnlyTrans(symOpInvariantsUp, basisStatesDown, self.latticeSize, kValue)

        H_down = self._calcHamiltonDownOnlyTrans(compDownStatesPerRepr, compInd2ReprDown, self.paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown, self.latticeSize)
        H_up = self._calcHamiltonUpOnlyTrans( basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp, symOp2ReprUp, intStatesUp , self.latticeSize)
        H_diag = self._calcHubbardDIAG( basisReprUp, normHubbardStates, compDownStatesPerRepr, self.paramU)

        Hamk = H_diag + H_up + H_down
    
        #vals, vecs = eigs(Hamk, k=1, which = ‘SR’)
        vals, vecs = eigs(Hamk, k=1, which='SR')

        return [Qobj(Hamk), basisStatesUp, compDownStatesPerRepr, normHubbardStates]

    def _getNearestNeighbors(self, **kwargs):
        latticeType = kwargs["latticeType"]
        latticeSize = kwargs["latticeSize"]
    
        indNeighbors = np.arange(latticeSize[0])+ 1
        indNeighbors = np.roll(indNeighbors,-1) - 1
        nSites = latticeSize[0]
        return [indNeighbors, nSites]

    def _ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    def createHeisenbergBasis(self, nStates, nSites, S_ges):
        basisStates = np.zeros((nStates, nSites))
        basisStates[0][nSites-S_ges: nSites] = 1
    
        iSites = np.arange(1, nSites+1,1)
        indOnes = np.zeros((nStates, S_ges))
    
        for iStates in range(0, nStates-1):
            maskOnes = basisStates[iStates, :] == 1
            indexOnes = iSites[maskOnes]
            indOnes[iStates][:] = indexOnes
            indexZeros = iSites[np.invert(maskOnes)]
        
            rightMostZero = max(indexZeros[ indexZeros < max(indexOnes)])-1
            basisStates[iStates+1, :] = basisStates[iStates, :]
            basisStates[iStates+1, rightMostZero] = 1
            basisStates[iStates+1, rightMostZero+1:nSites] = 0
        
            nDeletedOnes = sum( indexOnes > rightMostZero+1 )
            basisStates[iStates+1, nSites-nDeletedOnes+1:nSites] = 1        
        
        indOnes[nStates-1][:] = iSites[basisStates[nStates-1, :] == 1]    
    
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)

        integerBasis = np.sum(basisStates * bi2de, axis=1)
    
        return [basisStates, integerBasis, indOnes]

    def _cubicSymmetries(self, basisStates, symmetryValue, latticeSize):
        [nBasisStates, nSites] = np.shape(basisStates)
        symmetryPhases = np.zeros(nBasisStates)
        nSitesY = latticeSize[0]
    
        if symmetryValue == 0:
            symmetryPhases = np.power(1, symmetryPhases)
    
        if symmetryValue != 0:
            switchIndex = np.arange(1, nSites+1, 1)
            switchIndex = np.roll(switchIndex, symmetryValue, axis=0)
            basisStatesCopy = copy.deepcopy(basisStates)
            basisStatesCopy[:, 0:nSites] = basisStatesCopy[:, switchIndex-1]   
            flippedSwitchIndex = switchIndex[::-1]

            for iSites in range(1,nSites+1):
                indSwitchIndSmallerSiteInd = flippedSwitchIndex[flippedSwitchIndex <= iSites]
                indUpTo = np.arange(0, np.size(indSwitchIndSmallerSiteInd), 1)
            
                y = indSwitchIndSmallerSiteInd[ np.arange(0,indUpTo[indSwitchIndSmallerSiteInd==iSites],1) ]
                x = np.sum( basisStates[:, y-1], axis=1)
                r = basisStates[:, iSites-1]
                n = np.multiply(r, x)
                symmetryPhases = symmetryPhases + n

            symmetryPhases = np.power(-1, symmetryPhases)    

        if symmetryValue == 0:
            return [basisStates, symmetryPhases]
        else:
            return [basisStatesCopy, symmetryPhases]


    def _findReprOnlyTrans(self, basisStates, latticeSize):
        [nBasisStates, nSites] = np.shape(basisStates)
        latticeDim = np.shape(latticeSize)
        latticeDim = latticeDim[0]
    
        nSitesY = latticeSize[0]
        nSymmetricStates = nBasisStates * nSites
        indTransPower = np.arange(0,nSites,1)
    
        symmetricBasis = np.zeros((nSymmetricStates, nSites))
        symmetryPhases = np.ones(nSymmetricStates)
    
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)
    
        integerRepr = np.zeros(nBasisStates)
        index2Repr = np.zeros(nBasisStates)
        symOp2Repr = np.zeros(nBasisStates)
        basisRepr = np.zeros((nBasisStates, nSites))
    
        nRepr = 0
    
        symOpInvariants = np.zeros((nBasisStates, nSites))
    
        for rx1D in range(nSites):
            ind = rx1D + 1
            indCycle = np.arange(ind, nSymmetricStates - nSites + ind+1, nSites) - 1

            [symmetricBasis[indCycle, :] , symmetryPhases[indCycle] ] = self._cubicSymmetries( basisStates, rx1D, latticeSize)

        for iBasis in range(nBasisStates):
            # index to pick out one spec. symmetry operation
            indSymmetry = np.arange(iBasis*nSites, (iBasis+1)*nSites, 1)
            #pick binary form of one symmetry op. and calculate integer

            specSymBasis = symmetricBasis[indSymmetry, :]
        
            specInteger = np.sum(specSymBasis*bi2de, axis=1)
            specPhases = symmetryPhases[indSymmetry]
        
            # find unique integers
            [uniqueInteger, indUniqueInteger, conversionIndex2UniqueInteger] = np.unique(specInteger, return_index=True, return_inverse=True)

            if uniqueInteger[0] in integerRepr:
                locs = np.argwhere(integerRepr == uniqueInteger[0])
                alreadyRepr = locs[0][0]
                #position of corresponding repr. times phase factor
                index2Repr[iBasis] = alreadyRepr*specPhases[indUniqueInteger[0]] + 1
                # symmetry operations needed to get there, for 1D easy: position in
                # symmetry group-1, for 2D only translation too : rx*Ly + ry
                symOp2Repr[iBasis] = indUniqueInteger[0] 
            else:
                # integer value of repr. (needed in the loop)
                integerRepr[nRepr] = uniqueInteger[0]
                # binary repr. (output) if its a repr. its always first element
                # since list is sorted!:
                basisRepr[nRepr][:] = specSymBasis[0][:]
                # mask for same element as starting state
                sameElementAsFirst = conversionIndex2UniqueInteger == 0
            
                # store phases and translation value for invariant states
                # column position of non zero elements determines the combination
                # of translational powers rx, and ry: columnPos = rx*Ly + ry + 1
                symOpInvariants[nRepr][indTransPower[sameElementAsFirst]] = specPhases[sameElementAsFirst]
                # save index for hash table connecting basis states and repr.
                index2Repr[iBasis] = nRepr + 1
                # increase index for every found comp. repr.
                nRepr = nRepr+1
    
        # cut not used elements of container
        basisRepr = np.delete(basisRepr, np.arange(nRepr, nBasisStates, 1), 0)
        symOpInvariants = np.delete(symOpInvariants, np.arange(nRepr, nBasisStates, 1), 0)

        return [basisRepr, symOpInvariants, index2Repr, symOp2Repr]

    def _combine2HubbardBasisOnlyTrans(self, symOpInvariantsUp, basisStatesDown, latticeSize, kValue):
        [nReprUp, dumpV] = np.shape(symOpInvariantsUp)
        [nBasisStatesDown, nSites] = np.shape(basisStatesDown)
        [latticeDim, ] = np.shape(latticeSize)
        nSitesY = latticeSize[0]

        # umrechnung bin2dez
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)

        intDownStates = np.sum(basisStatesDown*bi2de, axis=1)
        indexDownStates = np.arange(1, nBasisStatesDown+1, 1)

        downStatesPerRepr = {}
        index2ReprDown = {}
        normHubbardStates = {}
    
    
        flagAlreadySaved = 0
        for iReprUp in range(nReprUp):
            transStates = 0
            transPhases = 0
            expPhase = 0
            transIndexUpT = np.argwhere(symOpInvariantsUp[iReprUp, :])
            transIndexUp = transIndexUpT[:, 0]

            transPhasesUp = symOpInvariantsUp[iReprUp, transIndexUp]
            ntransIndexUp = np.size(transIndexUp)
    
            if ntransIndexUp == 1:
                downStatesPerRepr[iReprUp] = basisStatesDown
                index2ReprDown[iReprUp] = indexDownStates
                normHubbardStates[iReprUp] = np.ones(nBasisStatesDown)/ nSites
        
            else:            
                transIndexUp = np.delete(transIndexUp, np.arange(0, 1, 1), 0)
                transPhasesUp = np.delete(transPhasesUp, np.arange(0, 1, 1), 0)
                maskStatesSmaller = np.ones(nBasisStatesDown)
                sumPhases = np.ones(nBasisStatesDown, dtype=complex)
        
                translationPower = transIndexUp       
                transPowerY = np.mod(transIndexUp, nSitesY)

                for iTrans in range(0, ntransIndexUp-1, 1):
                    [transStates, transPhases] = self._cubicSymmetries(basisStatesDown, translationPower[iTrans], latticeSize)
                    expPhase = np.exp(1J * kValue* translationPower[iTrans])
                    intTransStates = np.sum( transStates*bi2de, axis = 1)                
                    DLT = np.argwhere(intDownStates <= intTransStates)
                    DLT = DLT[:, 0]
                    set1 = np.zeros(nBasisStatesDown)
                    set1[DLT] = 1                
                    maskStatesSmaller = np.logical_and(maskStatesSmaller,  set1)
                    DLT = np.argwhere(intDownStates == intTransStates)
                    sameStates = DLT[:, 0]
                    sumPhases[sameStates] = sumPhases[sameStates] + expPhase * transPhasesUp[iTrans] * transPhases[sameStates]

                specNorm = np.abs(sumPhases)/ nSites
                DLT = np.argwhere(specNorm > 1e-10)
                DLT = DLT[:, 0]            
                maskStatesComp = np.zeros(nBasisStatesDown)
                maskStatesComp[DLT] = 1            
                maskStatesComp = np.logical_and(maskStatesSmaller,  maskStatesComp)
                downStatesPerRepr[iReprUp] = basisStatesDown[maskStatesComp, :]
                index2ReprDown[iReprUp] = indexDownStates[maskStatesComp]
                normHubbardStates[iReprUp] = specNorm[maskStatesComp]

        return [downStatesPerRepr, index2ReprDown, normHubbardStates]

    def _calcHubbardDIAG(self, basisReprUp, normHubbardStates, compDownStatesPerRepr, paramU):
        [nReprUp, dumpV] = np.shape(basisReprUp)
        nHubbardStates = 0
        for k in range(nReprUp):
            nHubbardStates = nHubbardStates + np.size(normHubbardStates[k])
        # container for double occupancies
        doubleOccupancies ={}
        #loop over repr.
        for iRepr in range(nReprUp):    
            # pick out down states per up spin repr.
            specBasisStatesDown = compDownStatesPerRepr[iRepr]
            [nReprDown, dumpV] = np.shape(specBasisStatesDown)
            UpReprs = [basisReprUp[iRepr, :]]* nReprDown
            doubleOccupancies[iRepr] = np.sum( np.logical_and(UpReprs, compDownStatesPerRepr[iRepr]), axis = 1 )
            if iRepr == 0:
                doubleOccupanciesA = doubleOccupancies[iRepr]
            elif iRepr > 0:
                doubleOccupanciesA = np.concatenate((doubleOccupanciesA, doubleOccupancies[iRepr]))
        rowIs = np.arange(0,nHubbardStates, 1)    
        H_diag = sparse.csr_matrix( ( paramU*doubleOccupanciesA,  (rowIs, rowIs)), shape=(nHubbardStates, nHubbardStates))    
        return H_diag

    def _calcHamiltonDownOnlyTrans(self, compDownStatesPerRepr, compInd2ReprDown, paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,latticeSize):
        [nReprUp, nSites] = np.shape(symOpInvariantsUp)
        nSitesY = latticeSize[0]
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)

        nTranslations = np.zeros(nReprUp, dtype=int)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            nTranslations[iReprUp] = np.sum(rt, axis=0)
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL

        [nDims, ] = np.shape(indNeighbors)
        # number of basis states of whole down spin basis
        [nBasisSatesDown, dumpV] = np.shape(basisStatesDown)
        # final x and y indices and phases

        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStatesDown, B2)
        TwoA = np.argwhere(B2)
        xIndWholeBasis = TwoA[:, 0]
        d = TwoA[:, 1]

        f = indNeighbors[d]
        f1 = np.append(d, f, axis=0)    
        d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(B2)), axis=0)    
                                      
        F_i = basisStatesDown[np.sum(B2, axis=1) == 0, : ]
        F = np.sum(F_i*bin2dez, axis=1)
                                    
        B2 = basisStatesDown[xIndWholeBasis, :]
        d1 = sparse.csr_matrix( (np.ones(np.size(d1)) ,  (d1, f1)), shape=(np.max(d1)+1, nSites )      )
        phasesWholeBasis = self._HubbardPhase(B2,d1,d,f)
        d = np.logical_xor(d1.todense()  , B2 )
        prodd = d * np.reshape(bin2dez,(nSitesY,1))

        d = np.sum( prodd, axis=1)    
        d = np.squeeze(np.asarray(d))    
        f = np.append(d, F, axis=0)
        [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

        yIndWholeBasis = B2[np.arange(0,np.size(xIndWholeBasis),1) ]
        sums = 0
        cumulIndFinal = np.zeros(nReprUp+1, dtype=int)
        cumulIndFinal[0] = sums

        xFinal = {}
        yFinal = {}
        phasesFinal = {}
        for iReprUp in range(nReprUp):
            specNumOfTrans = nTranslations[iReprUp]
            if specNumOfTrans == 1:
                xFinal[iReprUp] = cumulIndex[iReprUp] + xIndWholeBasis
                yFinal[iReprUp] = cumulIndex[iReprUp] + yIndWholeBasis
                phasesFinal[iReprUp] = phasesWholeBasis
            
                sums = sums + np.shape(xIndWholeBasis)[0]
                cumulIndFinal[iReprUp+1] = sums            
                continue

            specInd2ReprDown = compInd2ReprDown[iReprUp] - 1
            nSpecStatesDown = np.size(specInd2ReprDown)

            indexTransform = np.zeros(nBasisSatesDown)
            indexTransform[specInd2ReprDown] = np.arange(1,nSpecStatesDown+1,1)

            xIndSpec = indexTransform[xIndWholeBasis]
            yIndSpec = indexTransform[yIndWholeBasis]

            nBLe = np.size(xIndWholeBasis)

            DLT = np.argwhere(xIndSpec != 0)
            DLT = DLT[:, 0]
            maskStartingStates = np.zeros(nBLe, dtype=bool)
            maskStartingStates[DLT] = 1
        
            DLT = np.argwhere(yIndSpec != 0)
            DLT = DLT[:, 0]
            mask_ynonzero = np.zeros(nBLe)
            mask_ynonzero[DLT] = 1
        
            maskCompatible = np.logical_and(maskStartingStates, mask_ynonzero)
        
            xIndOfReprDown = {}
            yIndOfReprDown = {}
            hoppingPhase = {}
            for iTrans in range(specNumOfTrans-1):
                xIndOfReprDown[iTrans+1] = xIndSpec[maskStartingStates]
                hoppingPhase[iTrans+1] = phasesWholeBasis[maskStartingStates]
        
            xIndSpec = xIndSpec[maskCompatible]
            yIndSpec = yIndSpec[maskCompatible]
            phasesSpec = phasesWholeBasis[maskCompatible]
        
            xIndOfReprDown[0] = xIndSpec
            yIndOfReprDown[0] = yIndSpec
            hoppingPhase[0] = phasesSpec
        
            specStatesDown = compDownStatesPerRepr[iReprUp]
            intSpecStatesDown = np.sum(specStatesDown * bin2dez, axis=1)
        
            hoppedStates = basisStatesDown[ yIndWholeBasis[maskStartingStates] ,: ]
        
            DLT = np.argwhere( symOpInvariantsUp[iReprUp, :] )
        
            translationPower = DLT[:, 0] - 1
            phasesFromTransInvariance = symOpInvariantsUp[iReprUp, DLT]
            phasesFromTransInvariance = phasesFromTransInvariance[:, 0]

            cumulIndOfRepr = np.zeros(specNumOfTrans+1, dtype=int)
            cumulIndOfRepr[0] = 0
            cumulIndOfRepr[1] = np.size(xIndSpec)
            sumIOR = np.size(xIndSpec)


            for iTrans in range(specNumOfTrans-1):
                specTrans = translationPower[iTrans+1]

                [transBasis, transPhases] = self._cubicSymmetries( hoppedStates, specTrans, latticeSize)
                expPhases = np.exp(1J * kValue* specTrans)        
                phaseUpSpinTransInv = phasesFromTransInvariance[iTrans+1]        
                PhaseF = expPhases * np.multiply(transPhases, phaseUpSpinTransInv)        
                hoppingPhase[iTrans+1] = np.multiply(hoppingPhase[iTrans+1], PhaseF)        
                intValues = np.sum(transBasis * bin2dez, axis=1)            
                maskInBasis = np.in1d(intValues, intSpecStatesDown)            
                intValues = intValues[maskInBasis]            
                xIndOfReprDown[iTrans+1] = xIndOfReprDown[iTrans+1][maskInBasis]
                hoppingPhase[iTrans+1] = hoppingPhase[iTrans+1][maskInBasis]
                F = np.setdiff1d( intSpecStatesDown, intValues)
                f = np.append(intValues, F)        
                [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

                yIndOfReprDown[iTrans+1] = B2[np.arange(0,np.size(xIndOfReprDown[iTrans+1] ),1) ] + 1
                sumIOR = sumIOR + np.size(xIndOfReprDown[iTrans+1])
                cumulIndOfRepr[iTrans+2] = sumIOR

            xIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
            yIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
            hoppingPhaseA = np.zeros(cumulIndOfRepr[-1], dtype=complex)
            for iTrans in range(specNumOfTrans):
                xIndOfReprDownA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = xIndOfReprDown[iTrans] - 1
                yIndOfReprDownA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = yIndOfReprDown[iTrans] - 1
                hoppingPhaseA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = hoppingPhase[iTrans]
                    
            xFinal[iReprUp] = cumulIndex[iReprUp] + xIndOfReprDownA
            yFinal[iReprUp] = cumulIndex[iReprUp] + yIndOfReprDownA
            phasesFinal[iReprUp] = hoppingPhaseA

            sums = sums + np.size(xIndOfReprDownA)
            cumulIndFinal[iReprUp+1] = sums        
        
        xFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
        yFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
        phasesFinalA = np.zeros(cumulIndFinal[-1], dtype=complex)
    
        for iReprUp in range(nReprUp):
            xFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = xFinal[iReprUp ] 
            yFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = yFinal[iReprUp ]
            phasesFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = phasesFinal[iReprUp ]        
        
        nHubbardStates = cumulIndex[-1]
        normHubbardStatesA = np.zeros(nHubbardStates)    
        for iReprUp in range(nReprUp):
            normHubbardStatesA[ cumulIndex[iReprUp]: cumulIndex[iReprUp +1] ] = normHubbardStates[iReprUp]
    
        normHubbardStates = np.multiply( np.sqrt(normHubbardStatesA[xFinalA]) , np.sqrt(normHubbardStatesA[yFinalA]))

        H_down_elems = -paramT / nSites* np.divide(phasesFinalA, normHubbardStates)
        H_down = sparse.csr_matrix( (  H_down_elems,  (xFinalA, yFinalA)), shape=(nHubbardStates, nHubbardStates))
    
        return (H_down+ H_down.transpose().conjugate())/2

    def _calcHamiltonUpOnlyTrans(self, basisReprUp, compDownStatesPerRepr, paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp, symOp2ReprUp, intStatesUp , latticeSize):
        [nReprUp, nSites] = np.shape(basisReprUp)
        nSitesY = latticeSize[0]
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)
        nTranslations = np.zeros(nReprUp, dtype=int)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            nTranslations[iReprUp] = np.sum(rt, axis=0)
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL
        transPhases2ReprUp = np.sign(index2ReprUp)
        index2ReprUp = np.abs(index2ReprUp)

        B2 = basisReprUp[:, indNeighbors]
        B2 = np.logical_xor(basisReprUp, B2)

        TwoA = np.argwhere(B2)
        xIndOfReprUp = TwoA[:, 0]
        d = TwoA[:, 1]

        f = indNeighbors[d]
        f1 = np.append(d, f, axis=0)
    
        d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(B2)), axis=0)    
                                      
        F_i = basisReprUp[np.sum(B2, axis=1) == 0, : ]
        F = np.sum(F_i*bin2dez, axis=1)                                    
        B2 = basisReprUp[xIndOfReprUp, :]

        d1 = sparse.csr_matrix( (np.ones(np.size(d1)) ,  (d1, f1)), shape=(np.max(d1)+1, nSites )      )
        hoppingPhase = self._HubbardPhase(B2,d1,d,f)
        d = np.logical_xor(d1.todense()  , B2 )
        prodd = d * np.reshape(bin2dez,(nSitesY,1))
        d = np.sum( prodd, axis=1)    
        d = np.squeeze(np.asarray(d))   
        f = np.append(d, F, axis=0)      
        F = np.setdiff1d( intStatesUp, f)
        f = np.append(f, F)    

        [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

        yIndOfCycleUp = B2[np.arange(0,np.size(xIndOfReprUp),1) ]
        yIndOfReprUp = index2ReprUp[yIndOfCycleUp] - 1
        yIndOfReprUp = yIndOfReprUp.astype('int')    
        symOp2ReprUp = symOp2ReprUp[yIndOfCycleUp]    
        combPhases = np.multiply(transPhases2ReprUp[yIndOfCycleUp], hoppingPhase)
        xIndHubbUp = cumulIndex[xIndOfReprUp]
        yIndHubbUp = cumulIndex[yIndOfReprUp]    
        nConnectedUpStates = np.size(xIndOfReprUp)
    
        xFinal = {}
        yFinal = {}
        phasesFinal = {}
        cumFinal = np.zeros(nConnectedUpStates+1)
        sumF = 0
        cumFinal[0] = sumF
    
        for iStates in range(nConnectedUpStates):
            stateIndex = yIndOfReprUp[iStates]
            downSpinState1 = compDownStatesPerRepr[ xIndOfReprUp[iStates] ] 
            downSpinState2 = compDownStatesPerRepr[ stateIndex ]
            DLT = np.argwhere( symOpInvariantsUp[stateIndex, :] )        
            translationPower = DLT[:, 0]
            phasesFromTransInvariance = symOpInvariantsUp[stateIndex, DLT]
            phasesFromTransInvariance = phasesFromTransInvariance[:, 0]
            combTranslation = symOp2ReprUp[iStates] + translationPower
        
            nTrans = np.size(combTranslation)                
            xInd = {}
            yInd = {}
            phaseFactor = {}        
            cumI = np.zeros( nTrans+1, dtype=int)
            sumE = 0
            cumI[0] = sumE
        
            for iTrans in range(  nTrans ):
                Tss = -combTranslation[iTrans]
                [transStates, transPhases] = self._cubicSymmetries(downSpinState2, int(Tss), latticeSize)

                intTransStates = np.sum( transStates * bin2dez, axis=1)
                intStatesOne = np.sum( downSpinState1 * bin2dez , axis = 1 )
            
                [dumpV, xInd[iTrans], yInd[iTrans] ] = self._intersect_mtlb(intStatesOne, intTransStates)
                phaseFactor[iTrans] = transPhases[ yInd[iTrans] ]

                sumE = sumE + np.size( xInd[iTrans] )
                cumI[iTrans+1] = sumE

            xIndA = np.zeros(cumI[-1])
            yIndA = np.zeros(cumI[-1])

            for iTrans in range( nTrans ):
                xIndA[ cumI[iTrans]: cumI[iTrans +1] ] = xInd[iTrans ] 
                yIndA[ cumI[iTrans]: cumI[iTrans +1] ] = yInd[iTrans ]

            xFinal[iStates] = xIndHubbUp[iStates] + xIndA
            yFinal[iStates] = yIndHubbUp[iStates] + yIndA
            specCombPhase = combPhases[iStates]
            cumP = np.zeros(nTranslations[stateIndex]+1, dtype=int)
            sump = 0
            cumP[0] = sump
        
            for iTrans in range(nTranslations[stateIndex]):
                phaseFromTransUp = phasesFromTransInvariance[iTrans]
                expPhases = np.exp(1J * kValue * combTranslation[iTrans])
                phaseFactor[iTrans] = phaseFactor[iTrans] * specCombPhase * phaseFromTransUp * expPhases
                sump = sump + 1
                cumP[iTrans+1] = sump
        
            if nTrans == 1:
                phasesFinal[iStates] = phaseFactor[0]
            
            else:
                phasesFinal[iStates] = phaseFactor[0]
            
                for noss in range(1,nTrans,1):
                    phasesFinal[iStates ] = np.hstack( [ phasesFinal[iStates], phaseFactor[noss] ]  )

        phasesFinalA = phasesFinal[0]
        xFinalA = xFinal[0]
        yFinalA = yFinal[0]

        if nConnectedUpStates > 1:
            for noss in range(1, nConnectedUpStates, 1):
                phasesFinalA = np.hstack([ phasesFinalA, phasesFinal[noss] ])
                xFinalA = np.hstack([ xFinalA, xFinal[noss] ])
                yFinalA = np.hstack([ yFinalA, yFinal[noss] ])
        
        normHubbardStatesA = normHubbardStates[0]
        for iReprUp in range(1, nReprUp , 1):
            normHubbardStatesA = np.hstack([ normHubbardStatesA , normHubbardStates[iReprUp] ])
        
        nHubbardStates = np.size(normHubbardStatesA)

            
        normHubbardStates = np.multiply( np.sqrt(normHubbardStatesA[xFinalA.astype(int)]) , np.sqrt(normHubbardStatesA[yFinalA.astype(int)] ) )  
        H_up_elems = -paramT / nSites* np.divide(phasesFinalA, normHubbardStates)
        H_up = sparse.csr_matrix( (  H_up_elems,  (xFinalA, yFinalA)), shape=(nHubbardStates, nHubbardStates))
    
        return (H_up+ H_up.transpose().conjugate() )/2

    def _intersect_mtlb(self, a, b):
        a1, ia = np.unique(a, return_index=True)
        b1, ib = np.unique(b, return_index=True)
        aux = np.concatenate((a1, b1))
        aux.sort()
        c = aux[:-1][aux[1:] == aux[:-1]]
        return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

    def _HubbardPhase(self, replBasisStates, hopIndicationMatrix, hoppingIndicesX, hoppingIndicesY):
        [nReplBasisStates, nSites] = np.shape(replBasisStates) # number of states and sites
        siteInd = np.arange(1, nSites+1, 1)
        phase = np.zeros(nReplBasisStates)    
        hubbardPhases = np.zeros(nReplBasisStates)   #container for phases
        
        for i in range(nReplBasisStates):
            [dumpV, x] = hopIndicationMatrix.getrow(i).nonzero()
            if np.size(x) == 0:
                phase[i] = 0
                continue
            ToPow = replBasisStates[i, np.arange(x[0]+1, x[1], 1)]        
            if np.size(ToPow):
                phase[i] = np.power(-1, np.sum(ToPow, axis=0)  )
            else:
                phase[i] = 1
    
        return phase

class Lattice1d_2c_hcb_Hubbard():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a
    specified number of unit cells in the crystal. It can return dispersion
    relationship, position operators, Hamiltonian in the position represention
    etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit
        cell.
    Hamiltonian_of_cell : qutip.Qobj
        The Hamiltonian of the unit cell.
    inter_hop : qutip.Qobj / list of Qobj
        The coupling between the unit cell at i and at (i+unit vector)

    Attributes
    ----------
    num_cell : int
        The number of unit cells in the crystal.
    cell_num_site : int
        The nuber of sites in a unit cell.
    length_for_site : int
        The length of the dimension per site of a unit cell.
    cell_tensor_config : list of int
        The tensor structure of the cell in the form
        [cell_num_site,cell_site_dof[:][0] ]
    lattice_tensor_config : list of int
        The tensor structure of the crystal in the
        form [num_cell,cell_num_site,cell_site_dof[:][0]]
    length_of_unit_cell : int
        The length of the dimension for a unit cell.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components
        along Cartesian unit vectors.
    H_intra : qutip.Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ qutip.Qobj
        The list of coupling terms between unit cells of the lattice.
    is_real : bool
        Indicates if the Hamiltonian is real or not.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1,
                 U=1, fillingUp=None, fillingDown=None, k=None):

        self.paramT = t
        self.paramU = U
        self.latticeSize = [num_sites]
        self.fillingUp = fillingUp
        self.fillingDown = fillingDown
        self.kval = k

        if (not isinstance(num_sites, int)) or num_sites > 18:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")
                            
        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_Hubbard object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n on-site interaction energy, U = " +
              str(self.U ) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def basis_hcbosonic_Hubbard_chain(self):
        """
        Produces a graphic portraying the lattice symbolically with a unit cell
        marked in it.

        Returns
        -------
        inter_T : Qobj
            The coefficient of $\psi_{i,N}^{\dagger}\psi_{0,i+1}$, i.e. the
            coupling between the two boundary sites of the two unit cells i and
            i+1.
        """

        latticeType = 'cubic'

        [indNeighbors, nSites] = self._getNearestNeighbors(latticeType=latticeType, latticeSize=self.latticeSize)

        nStatesUp = self._ncr(nSites, self.fillingUp)
        nStatesDown = self._ncr(nSites, self.fillingDown)

        linearLatticeSizeY = self.latticeSize[0]


        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/linearLatticeSizeY)
        [basisStatesUp, intStatesUp, indOnesUp] = self.createHeisenbergBasis(nStatesUp, nSites, self.fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown] = self.createHeisenbergBasis(nStatesDown, nSites, self.fillingDown)

        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp] = self._findReprOnlyTrans(basisStatesUp, self.latticeSize)
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)

        intDownStates = np.sum(basisStatesDown*bin2dez, axis=1)
        intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)


        kValue = kVector[self.kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates] = self._combine2HubbardBasisOnlyTrans(symOpInvariantsUp, basisStatesDown, self.latticeSize, kValue)


        H_down = self._calcHamiltonDownOnlyTrans( compDownStatesPerRepr, compInd2ReprDown, self.paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown, self.latticeSize)
        H_up = self._calcHamiltonUpOnlyTrans( basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp, symOp2ReprUp, intStatesUp , self.latticeSize)
        H_diag = self._calcHubbardDIAG( basisReprUp, normHubbardStates, compDownStatesPerRepr, self.paramU)

        Hamk = H_diag + H_up + H_down
    
        #vals, vecs = eigs(Hamk, k=1, which = ‘SR’)
        vals, vecs = eigs(Hamk, k=1, which='SR')

        return [Qobj(Hamk), basisStatesUp, compDownStatesPerRepr, normHubbardStates]

    def _getNearestNeighbors(self, **kwargs):
        latticeType = kwargs["latticeType"]
        latticeSize = kwargs["latticeSize"]
    
        indNeighbors = np.arange(latticeSize[0])+ 1
        indNeighbors = np.roll(indNeighbors,-1) - 1
        nSites = latticeSize[0]
        return [indNeighbors, nSites]

    def _ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    def createHeisenbergBasis(self, nStates, nSites, S_ges):
        basisStates = np.zeros((nStates, nSites))
        basisStates[0][nSites-S_ges: nSites] = 1
    
        iSites = np.arange(1, nSites+1,1)
        indOnes = np.zeros((nStates, S_ges))
    
        for iStates in range(0, nStates-1):
            maskOnes = basisStates[iStates, :] == 1
            indexOnes = iSites[maskOnes]
            indOnes[iStates][:] = indexOnes
            indexZeros = iSites[np.invert(maskOnes)]
        
            rightMostZero = max(indexZeros[ indexZeros < max(indexOnes)])-1
            basisStates[iStates+1, :] = basisStates[iStates, :]
            basisStates[iStates+1, rightMostZero] = 1
            basisStates[iStates+1, rightMostZero+1:nSites] = 0
        
            nDeletedOnes = sum( indexOnes > rightMostZero+1 )
            basisStates[iStates+1, nSites-nDeletedOnes+1:nSites] = 1        
        
        indOnes[nStates-1][:] = iSites[basisStates[nStates-1, :] == 1]    
    
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)

        integerBasis = np.sum(basisStates * bi2de, axis=1)
    
        return [basisStates, integerBasis, indOnes]

    def _cubicSymmetries(self, basisStates, symmetryValue, latticeSize):
        [nBasisStates, nSites] = np.shape(basisStates)
        symmetryPhases = np.zeros(nBasisStates)
        nSitesY = latticeSize[0]
    
        if symmetryValue == 0:
            symmetryPhases = np.power(1, symmetryPhases)
    
        if symmetryValue != 0:
            switchIndex = np.arange(1, nSites+1, 1)
            switchIndex = np.roll(switchIndex, symmetryValue, axis=0)
            basisStatesCopy = copy.deepcopy(basisStates)
            basisStatesCopy[:, 0:nSites] = basisStatesCopy[:, switchIndex-1]   
            flippedSwitchIndex = switchIndex[::-1]

            for iSites in range(1,nSites+1):
                indSwitchIndSmallerSiteInd = flippedSwitchIndex[flippedSwitchIndex <= iSites]
                indUpTo = np.arange(0, np.size(indSwitchIndSmallerSiteInd), 1)
            
                y = indSwitchIndSmallerSiteInd[ np.arange(0,indUpTo[indSwitchIndSmallerSiteInd==iSites],1) ]
                x = np.sum( basisStates[:, y-1], axis=1)
                r = basisStates[:, iSites-1]
                n = np.multiply(r, x)
                symmetryPhases = symmetryPhases + n

            symmetryPhases = np.power(-1, symmetryPhases)    

        if symmetryValue == 0:
            return [basisStates, symmetryPhases]
        else:
            return [basisStatesCopy, symmetryPhases]


    def _findReprOnlyTrans(self, basisStates, latticeSize):
        [nBasisStates, nSites] = np.shape(basisStates)
        latticeDim = np.shape(latticeSize)
        latticeDim = latticeDim[0]
    
        nSitesY = latticeSize[0]
        nSymmetricStates = nBasisStates * nSites
        indTransPower = np.arange(0,nSites,1)
    
        symmetricBasis = np.zeros((nSymmetricStates, nSites))
        symmetryPhases = np.ones(nSymmetricStates)
    
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)
    
        integerRepr = np.zeros(nBasisStates)
        index2Repr = np.zeros(nBasisStates)
        symOp2Repr = np.zeros(nBasisStates)
        basisRepr = np.zeros((nBasisStates, nSites))
    
        nRepr = 0
    
        symOpInvariants = np.zeros((nBasisStates, nSites))
    
        for rx1D in range(nSites):
            ind = rx1D + 1
            indCycle = np.arange(ind, nSymmetricStates - nSites + ind+1, nSites) - 1

            [symmetricBasis[indCycle, :] , symmetryPhases[indCycle] ] = self._cubicSymmetries( basisStates, rx1D, latticeSize)

        for iBasis in range(nBasisStates):
            # index to pick out one spec. symmetry operation
            indSymmetry = np.arange(iBasis*nSites, (iBasis+1)*nSites, 1)
            #pick binary form of one symmetry op. and calculate integer

            specSymBasis = symmetricBasis[indSymmetry, :]
        
            specInteger = np.sum(specSymBasis*bi2de, axis=1)
            specPhases = symmetryPhases[indSymmetry]
        
            # find unique integers
            [uniqueInteger, indUniqueInteger, conversionIndex2UniqueInteger] = np.unique(specInteger, return_index=True, return_inverse=True)

            if uniqueInteger[0] in integerRepr:
                locs = np.argwhere(integerRepr == uniqueInteger[0])
                alreadyRepr = locs[0][0]
                #position of corresponding repr. times phase factor
                index2Repr[iBasis] = alreadyRepr*specPhases[indUniqueInteger[0]] + 1
                # symmetry operations needed to get there, for 1D easy: position in
                # symmetry group-1, for 2D only translation too : rx*Ly + ry
                symOp2Repr[iBasis] = indUniqueInteger[0] 
            else:
                # integer value of repr. (needed in the loop)
                integerRepr[nRepr] = uniqueInteger[0]
                # binary repr. (output) if its a repr. its always first element
                # since list is sorted!:
                basisRepr[nRepr][:] = specSymBasis[0][:]
                # mask for same element as starting state
                sameElementAsFirst = conversionIndex2UniqueInteger == 0
            
                # store phases and translation value for invariant states
                # column position of non zero elements determines the combination
                # of translational powers rx, and ry: columnPos = rx*Ly + ry + 1
                symOpInvariants[nRepr][indTransPower[sameElementAsFirst]] = specPhases[sameElementAsFirst]
                # save index for hash table connecting basis states and repr.
                index2Repr[iBasis] = nRepr + 1
                # increase index for every found comp. repr.
                nRepr = nRepr+1
    
        # cut not used elements of container
        basisRepr = np.delete(basisRepr, np.arange(nRepr, nBasisStates, 1), 0)
        symOpInvariants = np.delete(symOpInvariants, np.arange(nRepr, nBasisStates, 1), 0)

        return [basisRepr, symOpInvariants, index2Repr, symOp2Repr]

    def _combine2HubbardBasisOnlyTrans(self, symOpInvariantsUp, basisStatesDown, latticeSize, kValue):
        [nReprUp, dumpV] = np.shape(symOpInvariantsUp)
        [nBasisStatesDown, nSites] = np.shape(basisStatesDown)
        [latticeDim, ] = np.shape(latticeSize)
        nSitesY = latticeSize[0]

        # umrechnung bin2dez
        bi2de = np.arange(nSites-1,-1,-1)
        bi2de = np.power(2, bi2de)

        intDownStates = np.sum(basisStatesDown*bi2de, axis=1)
        indexDownStates = np.arange(1, nBasisStatesDown+1, 1)

        downStatesPerRepr = {}
        index2ReprDown = {}
        normHubbardStates = {}
    
        flagAlreadySaved = 0
        for iReprUp in range(nReprUp):
            transStates = 0
            transPhases = 0
            expPhase = 0
            transIndexUpT = np.argwhere(symOpInvariantsUp[iReprUp, :])
            transIndexUp = transIndexUpT[:, 0]

            transPhasesUp = symOpInvariantsUp[iReprUp, transIndexUp]
            ntransIndexUp = np.size(transIndexUp)
    
            if ntransIndexUp == 1:
                downStatesPerRepr[iReprUp] = basisStatesDown
                index2ReprDown[iReprUp] = indexDownStates
                normHubbardStates[iReprUp] = np.ones(nBasisStatesDown)/ nSites
        
            else:            
                transIndexUp = np.delete(transIndexUp, np.arange(0, 1, 1), 0)
                transPhasesUp = np.delete(transPhasesUp, np.arange(0, 1, 1), 0)
                maskStatesSmaller = np.ones(nBasisStatesDown)
                sumPhases = np.ones(nBasisStatesDown, dtype=complex)
        
                translationPower = transIndexUp       
                transPowerY = np.mod(transIndexUp, nSitesY)

                for iTrans in range(0, ntransIndexUp-1, 1):
                    [transStates, transPhases] = self._cubicSymmetries(basisStatesDown, translationPower[iTrans], latticeSize)
                    expPhase = np.exp(1J * kValue* translationPower[iTrans])
                    intTransStates = np.sum( transStates*bi2de, axis = 1)                
                    DLT = np.argwhere(intDownStates <= intTransStates)
                    DLT = DLT[:, 0]
                    set1 = np.zeros(nBasisStatesDown)
                    set1[DLT] = 1                
                    maskStatesSmaller = np.logical_and(maskStatesSmaller,  set1)
                    DLT = np.argwhere(intDownStates == intTransStates)
                    sameStates = DLT[:, 0]
                    sumPhases[sameStates] = sumPhases[sameStates] + expPhase * transPhasesUp[iTrans] * transPhases[sameStates]

                specNorm = np.abs(sumPhases)/ nSites
                DLT = np.argwhere(specNorm > 1e-10)
                DLT = DLT[:, 0]            
                maskStatesComp = np.zeros(nBasisStatesDown)
                maskStatesComp[DLT] = 1            
                maskStatesComp = np.logical_and(maskStatesSmaller,  maskStatesComp)
                downStatesPerRepr[iReprUp] = basisStatesDown[maskStatesComp, :]
                index2ReprDown[iReprUp] = indexDownStates[maskStatesComp]
                normHubbardStates[iReprUp] = specNorm[maskStatesComp]

        return [downStatesPerRepr, index2ReprDown, normHubbardStates]

    def _calcHubbardDIAG(self, basisReprUp, normHubbardStates, compDownStatesPerRepr, paramU):
        [nReprUp, dumpV] = np.shape(basisReprUp)
        nHubbardStates = 0
        for k in range(nReprUp):
            nHubbardStates = nHubbardStates + np.size(normHubbardStates[k])
        # container for double occupancies
        doubleOccupancies ={}
        #loop over repr.
        for iRepr in range(nReprUp):    
            # pick out down states per up spin repr.
            specBasisStatesDown = compDownStatesPerRepr[iRepr]
            [nReprDown, dumpV] = np.shape(specBasisStatesDown)
            UpReprs = [basisReprUp[iRepr, :]]* nReprDown
            doubleOccupancies[iRepr] = np.sum( np.logical_and(UpReprs, compDownStatesPerRepr[iRepr]), axis = 1 )
            if iRepr == 0:
                doubleOccupanciesA = doubleOccupancies[iRepr]
            elif iRepr > 0:
                doubleOccupanciesA = np.concatenate((doubleOccupanciesA, doubleOccupancies[iRepr]))
        rowIs = np.arange(0,nHubbardStates, 1)    
        H_diag = sparse.csr_matrix( ( paramU*doubleOccupanciesA,  (rowIs, rowIs)), shape=(nHubbardStates, nHubbardStates))    
        return H_diag

    def _calcHamiltonDownOnlyTrans(self, compDownStatesPerRepr, compInd2ReprDown, paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,latticeSize):
        [nReprUp, nSites] = np.shape(symOpInvariantsUp)
        nSitesY = latticeSize[0]
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)

        nTranslations = np.zeros(nReprUp, dtype=int)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            nTranslations[iReprUp] = np.sum(rt, axis=0)
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL

        [nDims, ] = np.shape(indNeighbors)
        # number of basis states of whole down spin basis
        [nBasisSatesDown, dumpV] = np.shape(basisStatesDown)
        # final x and y indices and phases

        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStatesDown, B2)
        TwoA = np.argwhere(B2)
        xIndWholeBasis = TwoA[:, 0]
        d = TwoA[:, 1]

        f = indNeighbors[d]
        f1 = np.append(d, f, axis=0)    
        d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(B2)), axis=0)    
                                      
        F_i = basisStatesDown[np.sum(B2, axis=1) == 0, : ]
        F = np.sum(F_i*bin2dez, axis=1)
                                    
        B2 = basisStatesDown[xIndWholeBasis, :]
        d1 = sparse.csr_matrix( (np.ones(np.size(d1)) ,  (d1, f1)), shape=(np.max(d1)+1, nSites )      )
        phasesWholeBasis = self._HubbardPhase(B2,d1,d,f)
        d = np.logical_xor(d1.todense()  , B2 )
        prodd = d * np.reshape(bin2dez,(nSitesY,1))

        d = np.sum( prodd, axis=1)    
        d = np.squeeze(np.asarray(d))    
        f = np.append(d, F, axis=0)
        [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

        yIndWholeBasis = B2[np.arange(0,np.size(xIndWholeBasis),1) ]
        sums = 0
        cumulIndFinal = np.zeros(nReprUp+1, dtype=int)
        cumulIndFinal[0] = sums

        xFinal = {}
        yFinal = {}
        phasesFinal = {}
        for iReprUp in range(nReprUp):
            specNumOfTrans = nTranslations[iReprUp]
            if specNumOfTrans == 1:
                xFinal[iReprUp] = cumulIndex[iReprUp] + xIndWholeBasis
                yFinal[iReprUp] = cumulIndex[iReprUp] + yIndWholeBasis
                phasesFinal[iReprUp] = phasesWholeBasis
            
                sums = sums + np.shape(xIndWholeBasis)[0]
                cumulIndFinal[iReprUp+1] = sums            
                continue

            specInd2ReprDown = compInd2ReprDown[iReprUp] - 1
            nSpecStatesDown = np.size(specInd2ReprDown)

            indexTransform = np.zeros(nBasisSatesDown)
            indexTransform[specInd2ReprDown] = np.arange(1,nSpecStatesDown+1,1)

            xIndSpec = indexTransform[xIndWholeBasis]
            yIndSpec = indexTransform[yIndWholeBasis]

            nBLe = np.size(xIndWholeBasis)

            DLT = np.argwhere(xIndSpec != 0)
            DLT = DLT[:, 0]
            maskStartingStates = np.zeros(nBLe, dtype=bool)
            maskStartingStates[DLT] = 1
        
            DLT = np.argwhere(yIndSpec != 0)
            DLT = DLT[:, 0]
            mask_ynonzero = np.zeros(nBLe)
            mask_ynonzero[DLT] = 1
        
            maskCompatible = np.logical_and(maskStartingStates, mask_ynonzero)
        
            xIndOfReprDown = {}
            yIndOfReprDown = {}
            hoppingPhase = {}
            for iTrans in range(specNumOfTrans-1):
                xIndOfReprDown[iTrans+1] = xIndSpec[maskStartingStates]
                hoppingPhase[iTrans+1] = phasesWholeBasis[maskStartingStates]
        
            xIndSpec = xIndSpec[maskCompatible]
            yIndSpec = yIndSpec[maskCompatible]
            phasesSpec = phasesWholeBasis[maskCompatible]
        
            xIndOfReprDown[0] = xIndSpec
            yIndOfReprDown[0] = yIndSpec
            hoppingPhase[0] = phasesSpec
        
            specStatesDown = compDownStatesPerRepr[iReprUp]
            intSpecStatesDown = np.sum(specStatesDown * bin2dez, axis=1)
        
            hoppedStates = basisStatesDown[ yIndWholeBasis[maskStartingStates] ,: ]
        
            DLT = np.argwhere( symOpInvariantsUp[iReprUp, :] )
        
            translationPower = DLT[:, 0] - 1
            phasesFromTransInvariance = symOpInvariantsUp[iReprUp, DLT]
            phasesFromTransInvariance = phasesFromTransInvariance[:, 0]

            cumulIndOfRepr = np.zeros(specNumOfTrans+1, dtype=int)
            cumulIndOfRepr[0] = 0
            cumulIndOfRepr[1] = np.size(xIndSpec)
            sumIOR = np.size(xIndSpec)


            for iTrans in range(specNumOfTrans-1):
                specTrans = translationPower[iTrans+1]

                [transBasis, transPhases] = self._cubicSymmetries( hoppedStates, specTrans, latticeSize)
                expPhases = np.exp(1J * kValue* specTrans)        
                phaseUpSpinTransInv = phasesFromTransInvariance[iTrans+1]        
                PhaseF = expPhases * np.multiply(transPhases, phaseUpSpinTransInv)        
                hoppingPhase[iTrans+1] = np.multiply(hoppingPhase[iTrans+1], PhaseF)        
                intValues = np.sum(transBasis * bin2dez, axis=1)            
                maskInBasis = np.in1d(intValues, intSpecStatesDown)            
                intValues = intValues[maskInBasis]            
                xIndOfReprDown[iTrans+1] = xIndOfReprDown[iTrans+1][maskInBasis]
                hoppingPhase[iTrans+1] = hoppingPhase[iTrans+1][maskInBasis]
                F = np.setdiff1d( intSpecStatesDown, intValues)
                f = np.append(intValues, F)        
                [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

                yIndOfReprDown[iTrans+1] = B2[np.arange(0,np.size(xIndOfReprDown[iTrans+1] ),1) ] + 1
                sumIOR = sumIOR + np.size(xIndOfReprDown[iTrans+1])
                cumulIndOfRepr[iTrans+2] = sumIOR

            xIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
            yIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
            hoppingPhaseA = np.zeros(cumulIndOfRepr[-1], dtype=complex)
            for iTrans in range(specNumOfTrans):
                xIndOfReprDownA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = xIndOfReprDown[iTrans] - 1
                yIndOfReprDownA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = yIndOfReprDown[iTrans] - 1
                hoppingPhaseA[ cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans +1] ] = hoppingPhase[iTrans]
                    
            xFinal[iReprUp] = cumulIndex[iReprUp] + xIndOfReprDownA
            yFinal[iReprUp] = cumulIndex[iReprUp] + yIndOfReprDownA
            phasesFinal[iReprUp] = hoppingPhaseA

            sums = sums + np.size(xIndOfReprDownA)
            cumulIndFinal[iReprUp+1] = sums        
        
        xFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
        yFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
        phasesFinalA = np.zeros(cumulIndFinal[-1], dtype=complex)
    
        for iReprUp in range(nReprUp):
            xFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = xFinal[iReprUp ] 
            yFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = yFinal[iReprUp ]
            phasesFinalA[ cumulIndFinal[iReprUp]: cumulIndFinal[iReprUp +1] ] = phasesFinal[iReprUp ]        
        
        nHubbardStates = cumulIndex[-1]
        normHubbardStatesA = np.zeros(nHubbardStates)    
        for iReprUp in range(nReprUp):
            normHubbardStatesA[ cumulIndex[iReprUp]: cumulIndex[iReprUp +1] ] = normHubbardStates[iReprUp]
    
        normHubbardStates = np.multiply( np.sqrt(normHubbardStatesA[xFinalA]) , np.sqrt(normHubbardStatesA[yFinalA]))

        H_down_elems = -paramT / nSites* np.divide(phasesFinalA, normHubbardStates)
        H_down = sparse.csr_matrix( (  H_down_elems,  (xFinalA, yFinalA)), shape=(nHubbardStates, nHubbardStates))
    
        return (H_down+ H_down.transpose().conjugate())/2

    def _calcHamiltonUpOnlyTrans(self, basisReprUp, compDownStatesPerRepr, paramT, indNeighbors, normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp, symOp2ReprUp, intStatesUp , latticeSize):
        [nReprUp, nSites] = np.shape(basisReprUp)
        nSitesY = latticeSize[0]
        bin2dez = np.arange(nSites-1,-1,-1)
        bin2dez = np.power(2, bin2dez)
        nTranslations = np.zeros(nReprUp, dtype=int)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            nTranslations[iReprUp] = np.sum(rt, axis=0)
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL
        transPhases2ReprUp = np.sign(index2ReprUp)
        index2ReprUp = np.abs(index2ReprUp)

        B2 = basisReprUp[:, indNeighbors]
        B2 = np.logical_xor(basisReprUp, B2)

        TwoA = np.argwhere(B2)
        xIndOfReprUp = TwoA[:, 0]
        d = TwoA[:, 1]

        f = indNeighbors[d]
        f1 = np.append(d, f, axis=0)
    
        d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(B2)), axis=0)    
                                      
        F_i = basisReprUp[np.sum(B2, axis=1) == 0, : ]
        F = np.sum(F_i*bin2dez, axis=1)                                    
        B2 = basisReprUp[xIndOfReprUp, :]

        d1 = sparse.csr_matrix( (np.ones(np.size(d1)) ,  (d1, f1)), shape=(np.max(d1)+1, nSites )      )
        hoppingPhase = self._HubbardPhase(B2,d1,d,f)
        d = np.logical_xor(d1.todense()  , B2 )
        prodd = d * np.reshape(bin2dez,(nSitesY,1))
        d = np.sum( prodd, axis=1)    
        d = np.squeeze(np.asarray(d))   
        f = np.append(d, F, axis=0)      
        F = np.setdiff1d( intStatesUp, f)
        f = np.append(f, F)    

        [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True, return_inverse=True)    

        yIndOfCycleUp = B2[np.arange(0,np.size(xIndOfReprUp),1) ]
        yIndOfReprUp = index2ReprUp[yIndOfCycleUp] - 1
        yIndOfReprUp = yIndOfReprUp.astype('int')    
        symOp2ReprUp = symOp2ReprUp[yIndOfCycleUp]    
        combPhases = np.multiply(transPhases2ReprUp[yIndOfCycleUp], hoppingPhase)
        xIndHubbUp = cumulIndex[xIndOfReprUp]
        yIndHubbUp = cumulIndex[yIndOfReprUp]    
        nConnectedUpStates = np.size(xIndOfReprUp)
    
        xFinal = {}
        yFinal = {}
        phasesFinal = {}
        cumFinal = np.zeros(nConnectedUpStates+1)
        sumF = 0
        cumFinal[0] = sumF
    
        for iStates in range(nConnectedUpStates):
            stateIndex = yIndOfReprUp[iStates]
            downSpinState1 = compDownStatesPerRepr[ xIndOfReprUp[iStates] ] 
            downSpinState2 = compDownStatesPerRepr[ stateIndex ]
            DLT = np.argwhere( symOpInvariantsUp[stateIndex, :] )        
            translationPower = DLT[:, 0]
            phasesFromTransInvariance = symOpInvariantsUp[stateIndex, DLT]
            phasesFromTransInvariance = phasesFromTransInvariance[:, 0]
            combTranslation = symOp2ReprUp[iStates] + translationPower
        
            nTrans = np.size(combTranslation)                
            xInd = {}
            yInd = {}
            phaseFactor = {}        
            cumI = np.zeros( nTrans+1, dtype=int)
            sumE = 0
            cumI[0] = sumE
        
            for iTrans in range(  nTrans ):
                Tss = -combTranslation[iTrans]
                [transStates, transPhases] = self._cubicSymmetries(downSpinState2, int(Tss), latticeSize)

                intTransStates = np.sum( transStates * bin2dez, axis=1)
                intStatesOne = np.sum( downSpinState1 * bin2dez , axis = 1 )
            
                [dumpV, xInd[iTrans], yInd[iTrans] ] = self._intersect_mtlb(intStatesOne, intTransStates)
                phaseFactor[iTrans] = transPhases[ yInd[iTrans] ]

                sumE = sumE + np.size( xInd[iTrans] )
                cumI[iTrans+1] = sumE

            xIndA = np.zeros(cumI[-1])
            yIndA = np.zeros(cumI[-1])

            for iTrans in range( nTrans ):
                xIndA[ cumI[iTrans]: cumI[iTrans +1] ] = xInd[iTrans ] 
                yIndA[ cumI[iTrans]: cumI[iTrans +1] ] = yInd[iTrans ]

            xFinal[iStates] = xIndHubbUp[iStates] + xIndA
            yFinal[iStates] = yIndHubbUp[iStates] + yIndA
            specCombPhase = combPhases[iStates]
            cumP = np.zeros(nTranslations[stateIndex]+1, dtype=int)
            sump = 0
            cumP[0] = sump
        
            for iTrans in range(nTranslations[stateIndex]):
                phaseFromTransUp = phasesFromTransInvariance[iTrans]
                expPhases = np.exp(1J * kValue * combTranslation[iTrans])
                phaseFactor[iTrans] = phaseFactor[iTrans] * specCombPhase * phaseFromTransUp * expPhases
                sump = sump + 1
                cumP[iTrans+1] = sump
        
            if nTrans == 1:
                phasesFinal[iStates] = phaseFactor[0]
            
            else:
                phasesFinal[iStates] = phaseFactor[0]
            
                for noss in range(1,nTrans,1):
                    phasesFinal[iStates ] = np.hstack( [ phasesFinal[iStates], phaseFactor[noss] ]  )

        phasesFinalA = phasesFinal[0]
        xFinalA = xFinal[0]
        yFinalA = yFinal[0]

        if nConnectedUpStates > 1:
            for noss in range(1, nConnectedUpStates, 1):
                phasesFinalA = np.hstack([ phasesFinalA, phasesFinal[noss] ])
                xFinalA = np.hstack([ xFinalA, xFinal[noss] ])
                yFinalA = np.hstack([ yFinalA, yFinal[noss] ])
        
        normHubbardStatesA = normHubbardStates[0]
        for iReprUp in range(1, nReprUp , 1):
            normHubbardStatesA = np.hstack([ normHubbardStatesA , normHubbardStates[iReprUp] ])
        
        nHubbardStates = np.size(normHubbardStatesA)

            
        normHubbardStates = np.multiply( np.sqrt(normHubbardStatesA[xFinalA.astype(int)]) , np.sqrt(normHubbardStatesA[yFinalA.astype(int)] ) )  
        H_up_elems = -paramT / nSites* np.divide(phasesFinalA, normHubbardStates)
        H_up = sparse.csr_matrix( (  H_up_elems,  (xFinalA, yFinalA)), shape=(nHubbardStates, nHubbardStates))
    
        return (H_up+ H_up.transpose().conjugate() )/2

    def _intersect_mtlb(self, a, b):
        a1, ia = np.unique(a, return_index=True)
        b1, ib = np.unique(b, return_index=True)
        aux = np.concatenate((a1, b1))
        aux.sort()
        c = aux[:-1][aux[1:] == aux[:-1]]
        return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

    def _HubbardPhase(self, replBasisStates, hopIndicationMatrix, hoppingIndicesX, hoppingIndicesY):
        [nReplBasisStates, nSites] = np.shape(replBasisStates) # number of states and sites
        siteInd = np.arange(1, nSites+1, 1)
        phase = np.ones(nReplBasisStates)    
    
        return phase

class Lattice1d_Heisenberg():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a
    specified number of unit cells in the crystal. It can return dispersion
    relationship, position operators, Hamiltonian in the position represention
    etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit
        cell.
    Hamiltonian_of_cell : qutip.Qobj
        The Hamiltonian of the unit cell.
    inter_hop : qutip.Qobj / list of Qobj
        The coupling between the unit cell at i and at (i+unit vector)

    Attributes
    ----------
    num_cell : int
        The number of unit cells in the crystal.
    cell_num_site : int
        The nuber of sites in a unit cell.
    length_for_site : int
        The length of the dimension per site of a unit cell.
    cell_tensor_config : list of int
        The tensor structure of the cell in the form
        [cell_num_site,cell_site_dof[:][0] ]
    lattice_tensor_config : list of int
        The tensor structure of the crystal in the
        form [num_cell,cell_num_site,cell_site_dof[:][0]]
    length_of_unit_cell : int
        The length of the dimension for a unit cell.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components
        along Cartesian unit vectors.
    H_intra : qutip.Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ qutip.Qobj
        The list of coupling terms between unit cells of the lattice.
    is_real : bool
        Indicates if the Hamiltonian is real or not.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1,
                 U=1, fillingUp=None, fillingDown=None, k=None):

        self.paramT = t
        self.paramU = U
        self.latticeSize = [num_sites]
        self.fillingUp = fillingUp
        self.fillingDown = fillingDown

        if (not isinstance(num_sites, int)) or num_sites > 18:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")
                            
        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_Hubbard object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n on-site interaction energy, U = " +
              str(self.U ) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

