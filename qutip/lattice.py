# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
__all__ = ['Lattice1d','cell_structures']
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   ceil, copysign, cos, cosh, degrees, e, exp, expm1, fabs,
                   floor, fmod, frexp, hypot, isinf, isnan, ldexp, log, log10,
                   log1p, modf, pi, radians, sin, sinh, sqrt, tan, tanh, trunc)
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip import *
import numpy as np
from scipy.sparse.linalg import eigs

def Hamiltonian_2d(base_h, inter_hop_x, inter_hop_y, nx_units = 1, PBCx = 0, ny_units = 1, PBCy = 0, cell_num_site = 1, length_for_site = 1 ):            
    """
    Returns the Hamiltonian as a csr_matrix from the specified parameters for a
    2d space of sites in the x-major format.
    
    Parameters
    ==========
    base_h : numpy matrix
        The Hamiltonian matrix of the unit cell
    inter_hop_x : numpy matrix
        The matrix coupling between cell i to the one at cell i+\hat{x} 
    inter_hop_y : numpy matrix
        The matrix coupling between cell i to the one at cell i+\hat{y} 
    site_dof_config : list
        The list of two numbers, namely the number of sites and the nuber of
        degrees of freedom per site of the unit cell
    nx_units : numpy matrix
        The length of the crystal in the x direction in units of unit cell length
    PBCx : int
        The indicator of periodic(1)/hardwall(0) boundary condition along direction x
    ny_units : numpy matrix
        The length of the crystal in the x direction in units of unit cell length
    PBCy : int
        The indicator of periodic(1)/hardwall(0) boundary condition along direction y

    Returns
    -------
    Hamt : csr_matrix
        The 2d Hamiltonian matrix for the specified parameters.                      
    """    
    xi_len = len(inter_hop_x)
    (x0,y0) = np.shape(inter_hop_x[0])

    (x1,y1) = np.shape(inter_hop_y)
    (xx,yy) = np.shape(base_h)

    row_ind = np.array([]); col_ind = np.array([]);  data = np.array([]);

    NS = cell_num_site*length_for_site

    for i in range(nx_units):
        for j in range(ny_units):
            lin_RI = i + j* nx_units                
            for k in range(xx):
                for l in range(yy):                        
                    row_ind = np.append(row_ind,[lin_RI*NS+k])
                    col_ind = np.append(col_ind,[lin_RI*NS+l])
                    data = np.append(data,[base_h[k,l] ])

    for i in range(0,nx_units):
        for j in range(0,ny_units):
            lin_RI = i + j* nx_units;

            for m in range(xi_len):

                for k in range(x0):
                    for l in range(y0):
                        if (i>0):
                            row_ind = np.append(row_ind,[lin_RI*NS+k])                        
                            col_ind = np.append(col_ind,[(lin_RI-1)*NS+l])
                            data = np.append(data,[np.conj(inter_hop_x[m][l,k]) ]);
  
                for k in range(x0):
                    for l in range(y0):
                        if (i < (nx_units-1)):
                            row_ind = np.append(row_ind,[lin_RI*NS+k])
                            col_ind = np.append(col_ind,[(lin_RI+1)*NS+l])
                            data = np.append(data,[inter_hop_x[m][k,l] ])

            for k in range(x1):
                for l in range(y1):
                    if (j>0):
                        row_ind = np.append(row_ind,[lin_RI*NS+k])
                        col_ind = np.append(col_ind,[(lin_RI-nx_units)*NS+l])
                        data = np.append(data,[np.conj(inter_hop_y[l,k]) ])
  
            for k in range(x1):
                for l in range(y1):
                    if (j<(ny_units-1)):
                        row_ind = np.append(row_ind,[lin_RI*NS+k])
                        col_ind = np.append(col_ind,[(lin_RI+nx_units)*NS+l])
                        data = np.append(data,[inter_hop_y[k,l] ])
           
    M = nx_units*ny_units*NS             

    for i in range(0,nx_units):
        lin_RI = i;                                               
        if (PBCy == 1 and ny_units*length_for_site > 2):
            for k in range(x1):
                for l in range(y1):
                    (Aw,)=np.where(row_ind == (lin_RI*NS+k)  )
                    (Bw,)=np.where(col_ind == ( (lin_RI+(ny_units-1)*nx_units)*NS+l )  )                    
                    if ( len(np.intersect1d(Aw,Bw)  ) == 0 ):
                        row_ind = np.append(row_ind,[lin_RI*NS+k,   (lin_RI+(ny_units-1)*nx_units)*NS+l]);
                        col_ind = np.append(col_ind,[(lin_RI+(ny_units-1)*nx_units)*NS+l,  lin_RI*NS+k]);
                        data = np.append(data,[np.conj(inter_hop_y[l,k]),  inter_hop_y[l,k] ]);

    for j in range(0,ny_units):
        lin_RI = j;
        if (PBCx == 1 and nx_units*cell_num_site > 2):
            for k in range(x0):
                for l in range(y0):
                    (Aw,)=np.where(row_ind == (lin_RI*nx_units*NS+k)  )
                    (Bw,)=np.where(col_ind == (((lin_RI+1)*nx_units-1)*NS+l)   )                    
                    if ( len(np.intersect1d(Aw,Bw)  ) == 0 ):                    
                        for m in range(xi_len):
                            row_ind = np.append(row_ind,[lin_RI*nx_units*NS+k,  ((lin_RI+1)*nx_units-1)*NS+l ]);
                            col_ind = np.append(col_ind,[((lin_RI+1)*nx_units-1)*NS+l,  lin_RI*nx_units*NS+k]);
                            data = np.append(data,[np.conj(inter_hop_x[m][l,k]), inter_hop_x[m][l,k] ]);
            
    Hamt = csr_matrix((data, (row_ind, col_ind)), [M, M], dtype=np.complex )            
    return Hamt

def diag_a_matrix( H_k, calc_evecs = False):
    """
    Returns eigen-values and/or eigen-vectors of an input matrix.
    
    Parameters
    ==========
    H_k : numpy matrix
        The matrix to be diagonalized
            
    Returns
    -------
    vecs : numpy ndarray
        vecs[:,:] = [band index,:] = eigen-vector of band_index
    vals : numpy ndarray
        The diagonalized matrix                        
    """                 
    if np.max(H_k-H_k.T.conj())>1.0E-20:
        raise Exception("\n\nThe Hamiltonian matrix is not hermitian!")

    if calc_evecs == False: # calculate only the eigenvalues
        vals=np.linalg.eigvalsh(H_k.todense())
        return np.array(vals,dtype=float)
    else: # find eigenvalues and eigenvectors
        (vals, vecs)=np.linalg.eigh(H_k.todense())
        vecs=vecs.T    
        # now vecs[i,:] is eigenvector for vals[i,i]-th eigenvalue        
        return (vals,vecs)

def cell_structures( val_s = [], val_t = [], val_u = [], val_v = [], limit = []):
    """
    Returns two matrices cell_H and cell_T to help the user form the inputs for
    defining an instance of Lattice1d and Lattice2d classes. The two matrices
    are the intra and inter cell Hamiltonians with the tensor structure of the 
    specified site numbers and/or degrees of freedom defined by the user.
    
    Parameters
    ==========
    val_s : list of str
        The first list of str's specifying the sites/degrees of freedom in the
        unitcell
        
    val_t : list of str
        The second list of str's specifying the sites/degrees of freedom in the
        unitcell
        
    val_u : list of str
        The third list of str's specifying the sites/degrees of freedom in the
        unitcell
        
    val_v : list of str
        The fourth list of str's specifying the sites/degrees of freedom in the
        unitcell
        
    limit:  list
        Informs the user of exceeding the capacity of the cell_structures function.
                
    Returns
    -------
    cell_H : numpy ndarray
        tensor structure of the cell Hamiltonian elements
    cell_T : numpy ndarray
        tensor structure of the inter cell Hamiltonian elements                        
    """                 
    SN = len(val_s)
    TN = len(val_t)
    UN = len(val_u)
    VN = len(val_v)
    LM = len(limit)

    if ( SN*TN == 0 and SN != 0 ):    
        cell_H = [ [ [] for i in range(SN)] for j in range(SN) ]
        inter_cell_T = [ [ [] for i in range(SN)] for j in range(SN) ]

        for ir in range(SN):
            for ic in range(SN):
                sst = val_s[ir]+" H "+val_s[ic]
                cell_H[ir][ic] = "<" + sst + ">"
                inter_cell_T[ir][ic] = "<cell(i):"+ sst + ":cell(i+1) >"
    
    if ( SN*TN*UN == 0 and SN*TN != 0 ):    
        cell_H = [ [ [] for i in range(SN*TN)] for j in range(SN*TN) ]
        inter_cell_T = [ [ [] for i in range(SN*TN)] for j in range(SN*TN) ]

        for ir in range(SN):
            for jr in range(TN):
                for ic in range(SN):
                    for jc in range(TN):
                        sst = val_s[ir]+val_t[jr]+ " H "+val_s[ic]+val_t[jc]
                        cell_H[ir*TN+jr][ic*TN+jc] = "<" + sst + ">" 
                        inter_cell_T[ir*TN+jr][ic*TN+jc] = "<cell(i):" + sst + ":cell(i+1) >" 

    if ( SN*TN*UN*VN == 0 and SN*TN*UN != 0 ):    
        cell_H = [ [ [] for i in range(SN*TN*UN)] for j in range(SN*TN*UN) ]
        inter_cell_T = [ [ [] for i in range(SN*TN*UN)] for j in range(SN*TN*UN) ]

        for ir in range(SN):
            for jr in range(TN):
                for kr in range(UN):
                    for ic in range(SN):
                        for jc in range(TN):
                            for kc in range(UN):
                                sst = val_s[ir]+val_t[jr]+val_u[kr]+ " H "+val_s[ic]+val_t[jc]+val_u[kc]
                                cell_H[ir*TN*UN+jr*UN+kr][ic*TN*UN+jc*UN+kc] = "<" + sst + ">" 
                                inter_cell_T[ir*TN*UN+jr*UN+kr][ic*TN*UN+jc*UN+kc] = "<cell(i):" + sst + ":cell(i+1) >" 

    if ( SN*TN*UN*VN*LM == 0 and SN*TN*UN*VN != 0 ):    
        cell_H = [ [ [] for i in range(SN*TN*UN*VN)] for j in range(SN*TN*UN*VN) ]
        inter_cell_T = [ [ [] for i in range(SN*TN*UN*VN)] for j in range(SN*TN*UN*VN) ]

        for ir in range(SN):
            for jr in range(TN):
                for kr in range(UN):
                    for lr in range(VN):
                        for ic in range(SN):
                            for jc in range(TN):
                                for kc in range(UN):
                                    for lc in range(VN):
                                        sst = val_s[ir]+val_t[jr]+val_u[kr]+val_v[lr]+ " H "+val_s[ic]+val_t[jc]+val_u[kc]+val_v[lc]
                                        cell_H[ir*TN*UN*VN+jr*UN*VN+kr*VN+lr][ic*TN*UN*VN+jc*UN*VN+kc*VN+lc] = "<" + sst + ">" 
                                        inter_cell_T[ir*TN*UN*VN+jr*UN*VN+kr*VN+lr][ic*TN*UN*VN+jc*UN*VN+kc*VN+lc] = "<cell(i):" + sst + ":cell(i+1) >" 

    if ( SN*TN*UN*VN*LM != 0 ):
        print("The cell_structures() function can not handle more than 4 lists!")
        cell_H = []
        inter_cell_T = []
    return (cell_H,inter_cell_T)

class Lattice1d():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a specified
    number of unit cells in the crystal. It can return dispersion relationship,
    position operators, Hamiltonian in the positio represention etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit cell.        
    cell_Hamiltonian :  Qobj        
        The Hamiltonian of the unit cell.
    inter_hop : Qobj / list of Qobj
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
    PBCx : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components 
        along Cartesian unit vectors.
    H_intra : Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ Qobj
        The list of coupling terms between unit cells of the lattice.           
    is_consistent : bool
        Indicates the consistency/correctness of all the attributes together
        in the unit Lattice1d.
        
    Methods
    -------
    Hamiltonian()
        Hamiltonian of the crystal.
    basis()
        basis with the particle localized at a certain cell,site with degree of
        freedom.
    distribute_operator()
        Distributes an input operator over all the cells.
    x()
        Position operator for the crystal.
    operator_at_cells()
        Distributes an input operator over user specified cells .
    plot_dispersion()
        Plots dispersion relation of the crystal.
    get_dispersion()
        Returns the dispersion relation of the crystal.      
    """    
    def __init__(self, num_cell=10, boundary = "periodic", cell_num_site = 1, cell_site_dof = [1], cell_Hamiltonian = Qobj([[ ]]) , inter_hop = Qobj([[ ]])   ):

        self.num_cell = num_cell        
        self.cell_num_site = cell_num_site

        if ( type(cell_num_site) != int or cell_num_site < 0 ):
            raise Exception("\n\n cell_num_site is required to be a positive integer.")

        if ( type(cell_site_dof) == list  ):
            l_v = 1
            for i in range(len(cell_site_dof)):
                if ( type(cell_site_dof[i]) != int or cell_site_dof[i] < 0 ):
                    print('\n Unacceptable cell_site_dof list element at index: ',i)
                    raise Exception("\n\n Elements of cell_site_dof is required to be positive integers.")
                l_v = l_v * cell_site_dof[i]                    
            self.cell_site_dof = cell_site_dof

        elif ( type(cell_site_dof) == int  ):
            if ( cell_site_dof < 0 ):
                raise Exception("\n\n cell_site_dof is required to be a positive integer.")
            else:
                l_v = cell_site_dof
                self.cell_site_dof = [cell_site_dof]                
        else:
            raise Exception("\n\n cell_site_dof is required to be a positive integer or a list of positive integers.")
        self._length_for_site = l_v 
        self.cell_tensor_config = list(np.append(cell_num_site, cell_site_dof) ) 
        self.lattice_tensor_config = list(np.append(num_cell,self.cell_tensor_config ) )         
        self._length_of_unit_cell = self.cell_num_site*self._length_for_site

        if boundary == "periodic":
            self.PBCx = 1;
        elif (boundary == "aperiodic" or boundary == "hardwall" ):
            self.PBCx = 0;
        else:
            print("Error in boundary")
            raise Exception(" Only recognized bounday options are:\"periodic\",\"aperiodic\" and \"hardwall\" ");
            
        self._inter_vec_list = [[self.cell_num_site]]       
        self._lattice_vectors_list = [[1]]     #unit vectors        

        if ( cell_Hamiltonian.dims[1][0] == 0):
            siteH = np.diag(np.zeros(cell_num_site-1)-1,1)+np.diag(np.zeros(cell_num_site-1)-1,-1)
            cell_Hamiltonian = tensor( Qobj(siteH), qeye(self.cell_site_dof) )
            
        self._H_intra = cell_Hamiltonian
        
        if ( not isinstance( cell_Hamiltonian, qutip.qobj.Qobj ) ):            
            raise Exception("\n\n cell_Hamiltonian is required to be a Qobj.")        

        nSb = self._H_intra.shape       
        if ( isinstance(inter_hop,list) ):    

            for i in range(len(inter_hop) ):
                
                if ( not isinstance( inter_hop[i], qutip.qobj.Qobj ) ):
                    print("\ninter_hop[",i,"] is not a Qobj.")
                    raise Exception("All inter_hop list elements need to be Qobj's. \n")
                    nSi = inter_hop[i].shape

                    if (nSb != nSi):
                        print("\n inter_hop[",i,"] is not dimensionally incorrect.\n")
                        raise Exception("All inter_hop list elements need to \
                        have the same dimensionality as cell_Hamiltonian.")
                                    
            self._H_inter_list = inter_hop
        elif( not isinstance( inter_hop, qutip.qobj.Qobj ) ):
            raise Exception("\n\n inter_hop need to be a Qobj.")
        else:
            if ( inter_hop.dims[1][0] == 0):
                if (cell_num_site == 1):
                    siteT = Qobj([[-1]])
                else:
                    siteT = basis(cell_num_site,cell_num_site-1)*basis(cell_num_site,0).dag() + basis(cell_num_site,0)*basis(cell_num_site,cell_num_site-1).dag()
                inter_hop = tensor( Qobj(siteT), qeye(self.cell_site_dof) )                                
                nSi = inter_hop.shape
                if (nSb != nSi):
                    raise Exception("inter_hop is required to have the same \
                    dimensionality as cell_Hamiltonian.")                
                self._H_inter_list = [inter_hop]     

        self._is_consistent = self._checks(check_if_complete = True)

    def __repr__(self):
        s = ""
        s += ("Lattice1d object: " +
              "Number of cells = " + str(self.num_cell) +
              ",\nNumber of sites in the cell = " + str(self.cell_num_site) +
              ",\nDegrees of freedom per site = " + str(self.lattice_tensor_config[2:len(self.lattice_tensor_config)]) +              
              ",\nLattice tensor configuration = " + str(self.lattice_tensor_config) +
              ",\nbasis_Hamiltonian = " + str(self._H_intra) +              
              ",\ninter_hop = " + str(self._H_inter_list) +
              ",\ninter_hop = " + str(self.cell_tensor_config) +
              ", isherm = " + str(self.cell_tensor_config) +
              "\n")
        if (self.PBCx == 1):
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"            

        return s
    
    def _checks(self,check_if_complete = False):
        """
        All user inputs are checked in __init__() at the time of initialization.
        Here it is checked if all the different inputs are consistent with each
        other. If it is called with check_if_complete == True, it returns if 
        the Lattice1d instance is complete or not.
        Returns
        -------
        is_consistent : bool
            Returns True if all the attributes and parameters are consistent
            for a complete definition of Lattice1d and False otherwise.        
        """                 
        if (not isherm(self._H_intra) ):
            raise Exception(" cell_Hamiltonian is required to be Hermitian. ");

        nRi = self._H_intra.shape[0]        
        if ( nRi != self._length_of_unit_cell):
            stt = " cell_Hamiltonian is not dimensionally consistent with \
            cell_num_site and cell_site_dof. cell_structure() function can be \
            used to obtain a valid structure for the cell_Hamiltonian and \
            inter_hop."
            raise Exception("stt");            

        is_consistent = True
        return is_consistent        

    def Hamiltonian(self):        
        """
        Returns the lattice Hamiltonian for the instance of Lattice1d.

        Returns
        ----------
        vec_i : Qobj
            oper type Quantum object representing the lattice Hamiltonian.      
        """        
        Hamil = Hamiltonian_2d(self._H_intra, self._H_inter_list, self._H_inter_list[0], nx_units = self.num_cell, PBCx = self.PBCx, ny_units = 1, PBCy = 0, cell_num_site = self.cell_num_site, length_for_site = self._length_for_site)
        # inter_hop_y = self._H_inter_list[0] is a dummy argument, since ny_units is 1,
        # the inter_hop_y coupling is immaterial.
        return Qobj(Hamil, dims=[ self.lattice_tensor_config, self.lattice_tensor_config ] )    
      
    def basis(self, cell, site, dof_ind  ):
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
        vec_i : Qobj
            ket type Quantum object representing the localized particle.
        """          
        if ( not isinstance(cell,int) ):
            raise Exception("cell needs to be int in basis().")
        elif( cell >= self.num_cell ):
            raise Exception("cell needs to less than Lattice1d.num_cell")
            
        if ( not isinstance(site,int) ):
            raise Exception("site needs to be int in basis().")
        elif( site >= self.cell_num_site ):
            raise Exception("site needs to less than Lattice1d.cell_num_site.")

        if( isinstance(dof_ind,int) ):
            dof_ind = [dof_ind]

        if ( not isinstance(dof_ind,list) ):
            raise Exception("dof_ind i basis() needs to be an int or list of int")
            
        if ( shape(dof_ind) == shape(self.cell_site_dof) ):
            for i in range(len(dof_ind)):
                if (dof_ind[i] >= self.cell_site_dof[i]):
                    print("in basis(), dof_ind[",i,"] is required to be smaller than \
                          cell_num_site[",i,"]")
                    raise Exception("\n Problem with dof_ind in basis()!")            
        else:
            raise Exception("dof_ind in basis() needs to be of the same dimensions \
                            as cell_site_dof.")
                      
        doft = basis( self.cell_site_dof[0], dof_ind[0]  )                                
        for i in range(1,len(dof_ind)):
            doft = tensor(doft, basis(self.cell_site_dof[i],dof_ind[i] )  )            
        vec_i = tensor(basis(self.num_cell,cell),basis(self.cell_num_site,site ), doft )                
        
        return vec_i       
    
    def distribute_operator(self, op  ):
        """
        A function that returns an operator matrix that applies op to all the 
        cells in the 1d lattice 
        
        Parameters
        -------
        op : Qobj
            Quantum object representing the operator to be applied at all cells. 

        Returns
        ----------
        Qobj(op_H) : Qobj
            Quantum object representing the operator with op applied at all cells.            
        """                 
        nSb = self._H_intra.shape       
        if ( not isinstance( op, qutip.qobj.Qobj ) ):
            raise Exception("op in distribute_operator need to be Qobj's. \n")
        nSi = op.shape
        if (nSb != nSi):
            print("\n inter_hop[",i,"] is not dimensionally incorrect.\n")
            raise Exception("op in distribute_operstor() is required to \
            have the same dimensionality as cell_Hamiltonian.")                                            
        
        (xx,yy) = np.shape(op)
        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])
        
        NS = self._length_of_unit_cell
        nx_units = self._num_cell
        ny_units = 1
        for i in range(nx_units):
            for j in range(ny_units):
                lin_RI = i + j* nx_units;                
                for k in range(xx):
                    for l in range(yy):                        
                        row_ind = np.append(row_ind,[lin_RI*NS+k]);
                        col_ind = np.append(col_ind,[lin_RI*NS+l]);
                        data = np.append(data,[op[k,l] ]);        
                        
        M = nx_units*ny_units*NS     
        op_H = csr_matrix((data, (row_ind, col_ind)), [M, M], dtype=np.complex )                    
        return Qobj(op_H)       

    def x(self):
        """
        Returns the position operator. All degrees of freedom has the cell number
        at their correspondig entry in the position operator.

        Returns
        -------
        Qobj(xs) : Qobj
            The position operator.        
        """     
        nx = self.cell_num_site
        ne = self._length_for_site

        positions = [ (j/ne) for i in range(nx) for j in range(ne) ] #not used in the
                                                        # current definition of x
        R = np.kron( range(0,self.num_cell), [1 for i in range(nx*ne) ] )
        S = np.kron( [1 for i in range(self.num_cell) ], positions )
#        xs = np.diagflat(R+S)        # not used in the
                                      # current definition of x
        xs = np.diagflat(R)        
        return Qobj(xs)    
        
    def operator_at_cells(self, op, cells ):
        """        
        A function that returns an operator matrix that applies op to specific
        cells specified in the cells list
        
        Parameters
        ----------
        op : qobj
            Quantum object representing the operator to be applied at certain cells.

        cells: list of int
            The cells at which the operator op is to be applied.                

        Returns
        -------
        isherm : bool
            Returns the new value of isherm property.        
        """                
        if (not isinstance(cells,list) ):
            if (isinstance(cells,int)):
                cells = [cells]
            else:
                raise Exception("cells in operator_at_cells() need to be a list of int.")                
        else:
            for i in range(len(cells) ):
                if (not isinstance(cells[i],int) ):
                    print("cells[",i,"] is not an int!")
                    raise Exception("elements of cells is required to be int's.")        

        nSb = self._H_intra.shape       
        if ( not isinstance( op, qutip.qobj.Qobj ) ):
            raise Exception("op in operator_at_cells need to be Qobj's. \n")
        nSi = op.shape
        if (nSb != nSi):
            raise Exception("op in operstor_at_cells() is required to \
            have the same dimensionality as cell_Hamiltonian.")      

        (xx,yy) = np.shape(op)
        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])
        NS = self._length_of_unit_cell
        nx_units = self.num_cell
        ny_units = 1                
        for i in range(nx_units):
            for j in range(ny_units):
                lin_RI = i + j* nx_units;                
                if ((i in cells) and j==0):
                    for k in range(xx):
                        for l in range(yy):
                            row_ind = np.append(row_ind,[lin_RI*NS+k])
                            col_ind = np.append(col_ind,[lin_RI*NS+l])
                            data = np.append(data,[op[k,l] ])        
                else:
                    for k in range(xx):
                        row_ind = np.append(row_ind,[lin_RI*NS+k]);
                        col_ind = np.append(col_ind,[lin_RI*NS+k]);
                        data = np.append(data,[1]);         
        
        M = nx_units*ny_units*NS                
        op_H = csr_matrix((data, (row_ind, col_ind)), [M, M], dtype=np.complex )                    
        return Qobj(op_H) 

    def plot_dispersion(self):
        """
        Plots the dispersion relationship for the lattice with the specified 
        number of unit cells along with the dispersion of the infinte crystal.
        """   
        a = self.cell_num_site
        klist = [-np.pi/a,np.pi/a,101]        
        knlist = [-np.pi/a,np.pi/a,self.num_cell+1]                
        
        k_start = klist[0]; k_end = klist[1]; kpoints = klist[2]   
        kn_start = knlist[0]; kn_end = knlist[1]; knpoints = knlist[2]-1         
        NS = self._length_of_unit_cell        
        val_ks=np.zeros((NS,kpoints),dtype=float)
        val_kns=np.zeros((NS,knpoints),dtype=float)        
        kxA = np.zeros((kpoints,1),dtype=float)
        knxA = np.zeros((knpoints,1),dtype=float)        
        G0_H = self._H_intra
        
        k_1 = (kpoints-1)
        for ks in range(kpoints):
            kx = k_start + (ks*(k_end-k_start)/k_1)
            kxA[ks,0] = kx
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._inter_vec_list)):
                r_cos = self._inter_vec_list[m]            
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*exp(complex(0,kr_dotted))
                H_ka = H_ka + H_int + H_int.dag();
            
            H_k = csr_matrix(H_ka)
            vals = diag_a_matrix(H_k, calc_evecs = False)                                
            val_ks[:,ks] = vals[:]

        for ks in range(knpoints):
            kx = kn_start + (ks*(kn_end-kn_start)/knpoints)
            knxA[ks,0] = kx
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._inter_vec_list)):
                r_cos = self._inter_vec_list[m]            
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*exp(complex(0,kr_dotted))
                H_ka = H_ka + H_int + H_int.dag();            
            H_k = csr_matrix(H_ka)
            vals = diag_a_matrix(H_k, calc_evecs = False)                                
            val_kns[:,ks] = vals[:]

        fig, ax = subplots()
        for g in range(NS):
            ax.plot(kxA/pi, val_ks[g,:]);
        for g in range(NS):
            ax.plot(np.append(knxA,[kn_end])/pi,np.append(val_kns[g,:],val_kns[g,0]), 'bo');            
        ax.set_ylabel('Energy');
        ax.set_xlabel('$k_x(\pi/a)$');
        show(fig)
        fig.savefig('./Dispersion.pdf')

    
    def get_dispersion(self):
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
        a = self.cell_num_site      
        knlist = [-np.pi/a,np.pi/a,self.num_cell+1]                
          
        kn_start = knlist[0]; kn_end = knlist[1]; knpoints = knlist[2]-1         
        NS = self._length_of_unit_cell        
        val_kns=np.zeros((NS,knpoints),dtype=float)   
        knxA = np.zeros((knpoints,1),dtype=float)        
        G0_H = self._H_intra

        for ks in range(knpoints):
            kx = kn_start + (ks*(kn_end-kn_start)/knpoints)
            knxA[ks,0] = kx
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._inter_vec_list)):
                r_cos = self._inter_vec_list[m]            
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*exp(complex(0,kr_dotted))
                H_ka = H_ka + H_int + H_int.dag();            
            H_k = csr_matrix(H_ka)
            vals = diag_a_matrix(H_k, calc_evecs = False)                                
            val_kns[:,ks] = vals[:]

        return (knxA,val_kns)
    

