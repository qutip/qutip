"""
PETSc backend for HEOM RHS matrix assembly.

This module provides a PETSc-based builder for the HEOM right-hand side
matrix. It streams sparse operator blocks directly into a distributed
PETSc matrix, avoiding the memory overhead of collecting Python objects
before assembly.
"""

import numpy as np
from petsc4py import PETSc


__all__ = ["PETScGatherHEOMRHS"]


class _PETScRHS:
    """Wrapper around a PETSc.Mat that provides the interface expected
    by IntegratorPETSc and HEOMSolver.

    This bridges the gap between PETSc's distributed matrix and the
    QuTiP solver infrastructure.

    Parameters
    ----------
    mat : PETSc.Mat
        The assembled PETSc matrix representing the HEOM RHS.
    sys_size : int
        The size of the system density matrix (i.e. ``N**2`` for an
        ``N``-level system).
    """

    def __init__(self, mat, sys_size):
        self.mat = mat
        self.sys_size = sys_size

    def arguments(self, args):
        """Validate solver arguments.

        Parameters
        ----------
        args : dict
            Solver arguments. Must be empty for the PETSc backend.

        Raises
        ------
        NotImplementedError
            If any time-dependent arguments are supplied.
        """
        if args:
            raise NotImplementedError(
                "Time-dependent arguments are not supported "
                "with the PETSc backend."
            )

    def __getattr__(self, name):
        return getattr(self.mat, name)


class PETScGatherHEOMRHS:
    """A class for collecting elements of the right-hand side matrix
    of the HEOM and streaming them directly into a distributed PETSc
    matrix to avoid Python object memory overhead.

    This class follows the same builder interface as ``_GatherHEOMRHS``
    (the CSR backend) but assembles blocks directly into a distributed
    PETSc ``Mat``.

    Parameters
    ----------
    f_idx : callable
        A function ``f_idx(he_state) -> int`` that returns the integer
        index of a hierarchy state (i.e. an ADO label).
    block : int
        The size of a single ADO Liouvillian operator in the hierarchy
        (i.e. ``sup_shape``).
    nhe : int
        The number of ADOs in the hierarchy.
    """

    def __init__(self, f_idx, block, nhe):
        self._block_size = block
        self._n_blocks = nhe
        self._f_idx = f_idx

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
        self.mat.setSizes(
            ((local_size, global_size), (local_size, global_size))
        )
        self.mat.setType(PETSc.Mat.Type.MPIAIJ)

        # Preallocation estimate
        # A row in the HEOM matrix is coupled to itself (via L_sys) and
        # its parents/children.  Max nonzeros per row in a block is much
        # smaller than the full block size due to sparsity.  We estimate
        # at most 60 nonzeros in the diagonal portion and 60 in the
        # off-diagonal portion to prevent out-of-memory errors for large
        # blocks while avoiding reallocation overhead.
        d_nnz = min(local_size, 60)
        o_nnz = max(min(global_size - local_size, 60), 0)
        self.mat.setPreallocationNNZ((d_nnz, o_nnz))
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # -- MPI label distribution ------------------------------------------

    def get_local_labels(self, labels):
        """Return the subset of ADO *labels* owned by the current MPI rank.

        The labels are distributed across ranks using the same
        block-cyclic scheme that the PETSc matrix rows use, so each
        rank only builds the matrix rows it owns.

        Parameters
        ----------
        labels : array_like
            The full list of ADO labels (hierarchy states).

        Returns
        -------
        local_labels : array_like
            The slice of *labels* assigned to this MPI rank.
        """
        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()

        n_blocks = len(labels)
        n_local_blocks = n_blocks // size
        remainder = n_blocks % size

        if rank < remainder:
            start_block = rank * (n_local_blocks + 1)
            end_block = start_block + n_local_blocks + 1
        else:
            start_block = rank * n_local_blocks + remainder
            end_block = start_block + n_local_blocks

        return labels[start_block:end_block]

    # -- Operator insertion ----------------------------------------------

    def add_op(self, row_he, col_he, op):
        """Add a block operator into the PETSc matrix.

        The operator *op* is placed at the block position
        ``(row_he, col_he)`` in the HEOM matrix.

        Parameters
        ----------
        row_he : hashable
            The ADO label for the row block.
        col_he : hashable
            The ADO label for the column block.
        op : :class:`qutip.data.Data`
            The sparse operator to insert.
        """
        row_blk = self._f_idx(row_he)
        col_blk = self._f_idx(col_he)

        sp_csr = op.as_scipy().tocsr()
        for i in range(sp_csr.shape[0]):
            start, end = sp_csr.indptr[i], sp_csr.indptr[i + 1]
            if start < end:
                global_row = row_blk * self._block_size + i
                global_cols = (
                    col_blk * self._block_size + sp_csr.indices[start:end]
                )
                vals = sp_csr.data[start:end]
                self.mat.setValues(
                    [global_row], global_cols, vals,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )

    # -- Assembly --------------------------------------------------------

    def gather(self, L_sys=None):
        """Assemble the PETSc matrix, optionally adding the system
        Liouvillian on the diagonal blocks.

        Parameters
        ----------
        L_sys : :class:`qutip.coefficient.Coefficient`, optional
            The system Liouvillian.  If provided and time-independent,
            its value is added to every diagonal block of the HEOM
            matrix.

        Returns
        -------
        mat : PETSc.Mat
            The assembled distributed PETSc matrix.
        """
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
                    start, end = L_sys_csr.indptr[i], L_sys_csr.indptr[i + 1]
                    if start < end:
                        global_row = r_blk * self._block_size + i
                        global_cols = (
                            r_blk * self._block_size
                            + L_sys_csr.indices[start:end]
                        )
                        vals = L_sys_csr.data[start:end]
                        self.mat.setValues(
                            [global_row], global_cols, vals,
                            addv=PETSc.InsertMode.ADD_VALUES,
                        )

        self.mat.assemblyBegin()
        self.mat.assemblyEnd()
        return self.mat

    # -- Finalize --------------------------------------------------------

    def finalize(self, L_sys, sup_shape, n_ados):
        """Assemble the final PETSc matrix and wrap it in a
        :class:`_PETScRHS` object.

        This method is the PETSc counterpart of the CSR backend's
        ``gather`` + ``QobjEvo`` wrapping step.  It validates that the
        system Liouvillian is time-independent, adds it to the diagonal
        blocks, assembles the distributed matrix, and returns a wrapper
        that satisfies the interface expected by the HEOM solver.

        Parameters
        ----------
        L_sys : :class:`qutip.coefficient.Coefficient`
            The system Liouvillian.
        sup_shape : int
            The superoperator dimension (``N**2`` for an ``N``-level
            system).
        n_ados : int
            The number of ADOs in the hierarchy.

        Returns
        -------
        rhs : :class:`_PETScRHS`
            A wrapper around the assembled PETSc matrix.

        Raises
        ------
        NotImplementedError
            If *L_sys* is time-dependent.
        """
        if not L_sys.isconstant:
            raise NotImplementedError(
                "PETSc backend currently supports only "
                "time-independent Liouvillians."
            )

        rhs_mat = self.gather(L_sys=L_sys)
        return _PETScRHS(rhs_mat, sup_shape)
