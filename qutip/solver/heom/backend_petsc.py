import numpy as np

class PETScGatherHEOMRHS:
    """ A class for collecting elements of the right-hand side matrix
        of the HEOM and streaming them directly into a distributed PETSc Matrix
        to avoid Python object memory overhead.
    """
    def __init__(self, f_idx, block, nhe):
        self._block_size = block
        self._n_blocks = nhe
        self._f_idx = f_idx
        
        try:
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("petsc4py is required for the PETSc backend.")
            
        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()
        
        global_size = block * nhe
        n_local_blocks = nhe // size
        remainder = nhe % size
        
        if rank < remainder:
            local_blocks = n_local_blocks + 1
        else:
            local_blocks = n_local_blocks
            
        local_size = local_blocks * block
        
        self.mat = PETSc.Mat().create(comm)
        self.mat.setSizes(((local_size, global_size), (local_size, global_size)))
        self.mat.setType(PETSc.Mat.Type.MPIAIJ)
        
        # Preallocation estimate
        # A row in the HEOM matrix is coupled to itself (via L_sys) and its parents/children.
        # Max nonzeros per row in a block is much smaller than the full block size due to sparsity.
        # We estimate at most 60 nonzeros in the diagonal portion and 60 in the off-diagonal portion
        # to prevent out of memory errors for large blocks while avoiding reallocation overhead.
        d_nnz = min(local_size, 60)
        o_nnz = min(global_size - local_size, 60)
        if o_nnz < 0: o_nnz = 0
        self.mat.setPreallocationNNZ((d_nnz, o_nnz))
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    def add_op(self, row_he, col_he, op):
        from petsc4py import PETSc
        row_blk = self._f_idx(row_he)
        col_blk = self._f_idx(col_he)
        
        row_indices = np.arange(row_blk * self._block_size, (row_blk + 1) * self._block_size, dtype=np.int32)
        col_indices = np.arange(col_blk * self._block_size, (col_blk + 1) * self._block_size, dtype=np.int32)
        
        sp_csr = op.as_scipy().tocsr()
        for i in range(sp_csr.shape[0]):
            start, end = sp_csr.indptr[i], sp_csr.indptr[i+1]
            if start < end:
                global_row = row_blk * self._block_size + i
                global_cols = col_blk * self._block_size + sp_csr.indices[start:end]
                vals = sp_csr.data[start:end]
                self.mat.setValues([global_row], global_cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

    def gather(self, L_sys=None):
        from petsc4py import PETSc
        if L_sys is not None and L_sys.isconstant:
            from qutip.core import data as _data
            L_sys_csr = _data.to(_data.CSR, L_sys(0).data).as_scipy()
            
            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()
            
            n_local_blocks = self._n_blocks // size
            remainder = self._n_blocks % size
            if rank < remainder:
                start_block = rank * (n_local_blocks + 1)
                end_block = start_block + n_local_blocks + 1
            else:
                start_block = rank * n_local_blocks + remainder
                end_block = start_block + n_local_blocks
            
            for r_blk in range(start_block, end_block):
                for i in range(L_sys_csr.shape[0]):
                    start, end = L_sys_csr.indptr[i], L_sys_csr.indptr[i+1]
                    if start < end:
                        global_row = r_blk * self._block_size + i
                        global_cols = r_blk * self._block_size + L_sys_csr.indices[start:end]
                        vals = L_sys_csr.data[start:end]
                        self.mat.setValues([global_row], global_cols, vals, addv=PETSc.InsertMode.ADD_VALUES)
                
        self.mat.assemblyBegin()
        self.mat.assemblyEnd()
        return self.mat
