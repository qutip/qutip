"""
Module contains functions for solving for the steady state density matrix of
open quantum systems defined by a Liouvillian or Hamiltonian and a list of
collapse operators.
"""

__all__ = ['steadystate', 'steadystate_floquet',
           'build_preconditioner', 'pseudo_inverse']

import warnings
import time

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.csgraph
import scipy.linalg
from scipy.sparse.linalg import (
    use_solver, splu, spilu, eigs, LinearOperator, gmres, lgmres, bicgstab,
)

from .. import (
    Qobj, liouvillian, unstack_columns, stack_columns, spre, tensor, identity,
    operator_to_vector,
)
from ..core import data as _data
from ..settings import settings
from . import _steadystate

import qutip.logging_utils
logger = qutip.logging_utils.get_logger()
logger.setLevel('DEBUG')

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import (mkl_splu, mkl_spsolve)


def _permute(matrix, rows, cols):
    """
    Permute the rows and columns of the scipy CSR or CSC matrix such that row
    `i` becomes row `rows[i]` and so on.  Returns the same type of CSR or CSC
    matrix.
    """
    # If the three data buffers of a CSC matrix are read as a CSR matrix (which
    # the corresponding shape change), it appears to be the transpose of the
    # input.  To handle CSC matrices here, we always use the data-layer type
    # CSR, but switch the rows and cols in the permutation if we actually want
    # a CSC, and take care to output the transpose of the transpose at the end.
    if not np.any(rows):
        rows = np.arange(matrix.shape[0])
    if not np.any(cols):
        cols = np.arange(matrix.shape[1])

    rows = np.argsort(rows)
    cols = np.argsort(cols)

    shape = matrix.shape
    if isinstance(matrix, scipy.sparse.csc_matrix):
        rows, cols = cols, rows
        shape = (shape[1], shape[0])
    temp = _data.CSR((matrix.data, matrix.indices, matrix.indptr),
                     shape=shape)
    temp = _data.permute.indices_csr(temp, rows, cols).as_scipy()
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return temp
    return scipy.sparse.csc_matrix((temp.data, temp.indices, temp.indptr),
                                   shape=matrix.shape)



def _profile(graph):
    profile = 0
    for row in range(graph.shape[0]):
        row_min = row_max = 0
        for ptr in range(graph.indptr[row], graph.indptr[row + 1]):
            if graph.data[ptr] == 0:
                continue
            dist = graph.indices[ptr] - row
            row_max = dist if dist > row_max else row_max
            row_min = dist if dist < row_min else row_min
        profile += row_max - row_min
    return profile


def _bandwidth(graph):
    upper = lower = 0
    for row in range(graph.shape[0]):
        for ptr in range(graph.indptr[row], graph.indptr[row + 1]):
            if graph.data[ptr] == 0:
                continue
            dist = graph.indices[ptr] - row
            lower = dist if dist < lower else lower
            upper = dist if dist > upper else upper
    return upper - lower + 1


def _weighted_bipartite_matching(A, perm_type='row'):
    """
    Returns an array of row permutations that attempts to maximize the product
    of the ABS values of the diagonal elements in a nonsingular square CSC
    sparse matrix. Such a permutation is always possible provided that the
    matrix is nonsingular.

    This function looks at both the structure and ABS values of the underlying
    matrix.

    Parameters
    ----------
    A : csc_matrix
        Input matrix

    perm_type : str {'row', 'column'}
        Type of permutation to generate.

    Returns
    -------
    perm : array
        Array of row or column permutations.

    Notes
    -----
    This function uses a weighted maximum cardinality bipartite matching
    algorithm based on breadth-first search (BFS).  The columns are weighted
    according to the element of max ABS value in the associated rows and are
    traversed in descending order by weight.  When performing the BFS
    traversal, the row associated to a given column is the one with maximum
    weight. Unlike other techniques[1]_, this algorithm does not guarantee the
    product of the diagonal is maximized.  However, this limitation is offset
    by the substantially faster runtime of this method.

    References
    ----------
    I. S. Duff and J. Koster, "The design and use of algorithms for permuting
    large entries to the diagonal of sparse matrices", SIAM J.  Matrix Anal.
    and Applics. 20, no. 4, 889 (1997).
    """
    nrows = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError('weighted_bfs_matching requires a square matrix.')
    if scipy.sparse.isspmatrix_csr(A) or scipy.sparse.isspmatrix_coo(A):
        A = A.tocsc()
    elif not scipy.sparse.isspmatrix_csc(A):
        raise TypeError("matrix must be in CSC, CSR, or COO format.")

    if perm_type == 'column':
        A = A.transpose().tocsc()
    perm = _steadystate.weighted_bipartite_matching(
                    np.asarray(np.abs(A.data), dtype=float),
                    A.indices, A.indptr, nrows)
    if np.any(perm == -1):
        raise Exception('Possibly singular input matrix.')
    return perm


def _empty_info_dict():
    def_info = {'perm': [], 'solution_time': None,
                'residual_norm': None,
                'solver': None, 'method': None}

    return def_info


def _default_steadystate_args():
    def_args = {'sparse': True, 'use_rcm': False,
                'use_wbm': False, 'use_precond': False,
                'all_states': False, 'M': None, 'x0': None, 'drop_tol': 1e-4,
                'fill_factor': 100, 'diag_pivot_thresh': 0.1, 'maxiter': 1000,
                'permc_spec': 'COLAMD', 'ILU_MILU': 'smilu_2',
                'restart': 20,
                'max_iter_refine': 10,
                'scaling_vectors': True,
                'weighted_matching': True,
                'return_info': False, 'info': _empty_info_dict(),
                'verbose': False, 'solver': 'scipy', 'weight': None,
                'tol': 1e-12, 'matol': 1e-15, 'mtol': None}
    return def_args


def steadystate(A, c_op_list=[], method='direct', solver=None, **kwargs):
    """
    Calculates the steady state for quantum evolution subject to the supplied
    Hamiltonian or Liouvillian operator and (if given a Hamiltonian) a list of
    collapse operators.

    If the user passes a Hamiltonian then it, along with the list of collapse
    operators, will be converted into a Liouvillian operator in Lindblad form.

    Parameters
    ----------
    A : :obj:`~Qobj`
        A Hamiltonian or Liouvillian operator.

    c_op_list : list
        A list of collapse operators.

    solver : {'scipy', 'mkl'}, optional
        Selects the sparse solver to use.  Default is to auto-select based on
        the availability of the MKL library.

    method : str, default 'direct'
        The allowed methods are
        - 'direct'
        - 'eigen'
        - 'iterative-gmres'
        - 'iterative-lgmres'
        - 'iterative-bicgstab'
        - 'svd'
        - 'power'
        - 'power-gmres'
        - 'power-lgmres'
        - 'power-bicgstab'

        Method for solving the underlying linear equation. Direct LU solver
        'direct' (default), sparse eigenvalue problem 'eigen', iterative GMRES
        method 'iterative-gmres', iterative LGMRES method 'iterative-lgmres',
        iterative BICGSTAB method 'iterative-bicgstab', SVD 'svd' (dense), or
        inverse-power method 'power'. The iterative power methods
        'power-gmres', 'power-lgmres', 'power-bicgstab' use the same solvers as
        their direct counterparts.

    return_info : bool, default False
        Return a dictionary of solver-specific infomation about the solution
        and how it was obtained.

    sparse : bool, default True
        Solve for the steady state using sparse algorithms. If set to False,
        the underlying Liouvillian operator will be converted into a dense
        matrix. Use only for 'smaller' systems.

    use_rcm : bool, default False
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the LU
        factorization of the Liouvillian.

    use_wbm : bool, default False
        Use Weighted Bipartite Matching reordering to make the Liouvillian
        diagonally dominant.  This is useful for iterative preconditioners
        only, and is set to ``True`` by default when finding a preconditioner.

    weight : float, optional
        Sets the size of the elements used for adding the unity trace condition
        to the linear solvers.  This is set to the average abs value of the
        Liouvillian elements if not specified by the user.

    max_iter_refine : int, default 10
        MKL ONLY. Max. number of iterative refinements to perform.

    scaling_vectors : bool
        MKL ONLY.  Scale matrix to unit norm columns and rows.

    weighted_matching : bool
        MKL ONLY.  Use weighted matching to better condition diagonal.

    x0 : ndarray, optional
        ITERATIVE ONLY. Initial guess for solution vector.

    maxiter : int, optional, default=1000
        ITERATIVE ONLY. Maximum number of iterations to perform.

    tol : float, default 1e-12
        ITERATIVE ONLY. Tolerance used for terminating solver.

    mtol : float, optional
        ITERATIVE 'power' methods ONLY. Tolerance for lu solve method.  If None
        given then ``max(0.1*tol, 1e-15)`` is used.

    matol : float, default 1e-15
        ITERATIVE ONLY. Absolute tolerance for lu solve method.

    permc_spec : str, optional
        ITERATIVE ONLY. Column ordering used internally by superLU for the
        'direct' LU decomposition method. Options include 'COLAMD' and
        'NATURAL'. If using RCM then this is set to 'NATURAL' automatically
        unless explicitly specified.

    use_precond : bool optional, default = False
        ITERATIVE ONLY. Use an incomplete sparse LU decomposition as a
        preconditioner for the 'iterative' GMRES and BICG solvers.
        Speeds up convergence time by orders of magnitude in many cases.

    M : {sparse matrix, dense matrix, LinearOperator}, optional
        ITERATIVE ONLY. Preconditioner for A. The preconditioner should
        approximate the inverse of A. Effective preconditioning can
        dramatically improve the rate of convergence for iterative methods.
        If no preconditioner is given and ``use_precond = True``, then one
        is generated automatically.

    fill_factor : float, default 100
        ITERATIVE ONLY. Specifies the fill ratio upper bound (>=1) of the iLU
        preconditioner.  Lower values save memory at the cost of longer
        execution times and a possible singular factorization.

    drop_tol : float, default 1e-4
        ITERATIVE ONLY. Sets the threshold for the magnitude of preconditioner
        elements that should be dropped.  Can be reduced for a courser
        factorization at the cost of an increased number of iterations, and a
        possible singular factorization.

    diag_pivot_thresh : float, optional
        ITERATIVE ONLY. Sets the threshold between [0,1] for which diagonal
        elements are considered acceptable pivot points when using a
        preconditioner.  A value of zero forces the pivot to be the diagonal
        element.

    ILU_MILU : str, default 'smilu_2'
        ITERATIVE ONLY. Selects the incomplete LU decomposition method algoithm
        used in creating the preconditoner. Should only be used by advanced
        users.

    Returns
    -------
    dm : qobj
        Steady state density matrix.
    info : dict, optional
        Dictionary containing solver-specific information about the solution.

    Notes
    -----
    The SVD method works only for dense operators (i.e. small systems).
    """
    if solver is None:
        solver = 'scipy'
        if settings.has_mkl:
            if method in ['direct', 'power']:
                solver = 'mkl'
    elif solver == 'mkl' and \
            (method not in ['direct', 'power']):
        raise ValueError('MKL solver only for direct or power methods.')

    elif solver not in ['scipy', 'mkl']:
        raise ValueError('Invalid solver kwarg.')

    ss_args = _default_steadystate_args()
    ss_args['method'] = method
    if solver is not None:
        ss_args['solver'] = solver
    ss_args['info']['solver'] = ss_args['solver']
    ss_args['info']['method'] = ss_args['method']

    for key, value in kwargs.items():
        if key in ss_args:
            ss_args[key] = value
        else:
            raise TypeError(
                "Invalid keyword argument '"+key+"' passed to steadystate.")

    # Set column perm to NATURAL if using RCM and not specified by user
    if ss_args['use_rcm'] and ('permc_spec' not in kwargs):
        ss_args['permc_spec'] = 'NATURAL'

    # Create & check Liouvillian
    A = _steadystate_setup(A, c_op_list)

    # Set weight parameter to max abs val in L if not set explicitly
    if 'weight' not in kwargs.keys():
        # set the weight to the mean of the non-zero absoluate values in A:
        A_np = np.abs(A.full())
        ss_args['weight'] = np.mean(A_np[A_np > 0])
        ss_args['info']['weight'] = ss_args['weight']

    if ss_args['method'] == 'direct':
        if (ss_args['solver'] == 'scipy' and ss_args['sparse']) \
                or ss_args['solver'] == 'mkl':
            return _steadystate_direct_sparse(A, ss_args)
        else:
            return _steadystate_direct_dense(A, ss_args)

    elif ss_args['method'] == 'eigen':
        return _steadystate_eigen(A, ss_args)

    elif ss_args['method'] in ['iterative-gmres',
                               'iterative-lgmres', 'iterative-bicgstab']:
        return _steadystate_iterative(A, ss_args)

    elif ss_args['method'] == 'svd':
        return _steadystate_svd_dense(A, ss_args)

    elif ss_args['method'] in ['power', 'power-gmres',
                               'power-lgmres', 'power-bicgstab']:
        return _steadystate_power(A, ss_args)

    else:
        raise ValueError('Invalid method argument for steadystate.')


def _steadystate_setup(A, c_op_list):
    """Build Liouvillian (if necessary) and check input."""
    if A.isoper:
        if len(c_op_list) > 0:
            return liouvillian(A, c_op_list)
        raise TypeError('Cannot calculate the steady state for a ' +
                        'non-dissipative system ' +
                        '(no collapse operators given)')
    if A.issuper:
        return A
    raise TypeError('Solving for steady states requires ' +
                    'Liouvillian (super) operators')


def _steadystate_LU_liouvillian(L, ss_args, has_mkl=0):
    """Creates modified Liouvillian for LU based SS methods.
    """
    perm = None
    perm2 = None
    rev_perm = None
    n = int(np.sqrt(L.shape[0]))
    L = _data.to(_data.CSR, L.data).as_scipy()
    if has_mkl:
        constructor = scipy.sparse.csr_matrix
    else:
        L = L.tocsc()
        constructor = scipy.sparse.csc_matrix
    L = L + constructor((ss_args['weight'] * np.ones(n),
                         (np.zeros(n), [nn * (n+1) for nn in range(n)])),
                        shape=(n*n, n*n))

    if settings.debug:
        old_band = _bandwidth(L)
        old_pro = _profile(L)
        logger.debug('Orig. NNZ: %i', L.nnz)
        if ss_args['use_rcm']:
            logger.debug('Original bandwidth: %i', old_band)

    if ss_args['use_wbm']:
        if settings.debug:
            logger.debug('Calculating Weighted Bipartite Matching ordering...')
        _wbm_start = time.time()
        perm = _weighted_bipartite_matching(L)
        _wbm_end = time.time()
        L = _permute(L, perm, None)
        ss_args['info']['perm'].append('wbm')
        ss_args['info']['wbm_time'] = _wbm_end-_wbm_start
        if settings.debug:
            wbm_band = _bandwidth(L)
            logger.debug('WBM bandwidth: %i' % wbm_band)

    if ss_args['use_rcm']:
        if settings.debug:
            logger.debug('Calculating Reverse Cuthill-Mckee ordering...')
        _rcm_start = time.time()
        perm2 = scipy.sparse.csgraph.reverse_cuthill_mckee(L)
        _rcm_end = time.time()
        rev_perm = np.argsort(perm2)
        L = _permute(L, perm2, perm2)
        ss_args['info']['perm'].append('rcm')
        ss_args['info']['rcm_time'] = _rcm_end-_rcm_start
        if settings.debug:
            rcm_band = _bandwidth(L)
            rcm_pro = _profile(L)
            logger.debug('RCM bandwidth: %i' % rcm_band)
            logger.debug('Bandwidth reduction factor: %f' %
                         (old_band/rcm_band))
            logger.debug('Profile reduction factor: %f' %
                         (old_pro/rcm_pro))
    L.sort_indices()
    return L, perm, perm2, rev_perm, ss_args


def _steadystate_direct_sparse(L, ss_args):
    """
    Direct solver that uses scipy sparse matrices
    """
    if settings.debug:
        logger.debug('Starting direct LU solver.')

    dims = L.dims[0]
    n = int(np.sqrt(L.shape[0]))
    b = np.zeros(n ** 2, dtype=complex)
    b[0] = ss_args['weight']

    if ss_args['solver'] == 'mkl':
        has_mkl = 1
    else:
        has_mkl = 0

    ss_lu_liouv_list = _steadystate_LU_liouvillian(L, ss_args, has_mkl)
    L, perm, perm2, rev_perm, ss_args = ss_lu_liouv_list
    if np.any(perm):
        b = b[np.ix_(perm,)]
    if np.any(perm2):
        b = b[np.ix_(perm2,)]

    if ss_args['solver'] == 'scipy':
        ss_args['info']['permc_spec'] = ss_args['permc_spec']
        ss_args['info']['drop_tol'] = ss_args['drop_tol']
        ss_args['info']['diag_pivot_thresh'] = ss_args['diag_pivot_thresh']
        ss_args['info']['fill_factor'] = ss_args['fill_factor']
        ss_args['info']['ILU_MILU'] = ss_args['ILU_MILU']

    if not ss_args['solver'] == 'mkl':
        # Use superLU solver
        orig_nnz = L.nnz
        _direct_start = time.time()
        lu = splu(L, permc_spec=ss_args['permc_spec'],
                  diag_pivot_thresh=ss_args['diag_pivot_thresh'],
                  options=dict(ILU_MILU=ss_args['ILU_MILU']))
        v = lu.solve(b)
        _direct_end = time.time()
        ss_args['info']['solution_time'] = _direct_end - _direct_start
        if (settings.debug or ss_args['return_info']):
            L_nnz = lu.L.nnz
            U_nnz = lu.U.nnz
            ss_args['info']['l_nnz'] = L_nnz
            ss_args['info']['u_nnz'] = U_nnz
            ss_args['info']['lu_fill_factor'] = (L_nnz + U_nnz)/L.nnz
            if settings.debug:
                logger.debug('L NNZ: %i ; U NNZ: %i' % (L_nnz, U_nnz))
                logger.debug('Fill factor: %f' % ((L_nnz + U_nnz)/orig_nnz))

    else:  # Use MKL solver
        if len(ss_args['info']['perm']) != 0:
            in_perm = np.arange(n**2, dtype=np.int32)
        else:
            in_perm = None
        _direct_start = time.time()
        v = mkl_spsolve(L, b, perm=in_perm, verbose=ss_args['verbose'],
                        max_iter_refine=ss_args['max_iter_refine'],
                        scaling_vectors=ss_args['scaling_vectors'],
                        weighted_matching=ss_args['weighted_matching'])
        _direct_end = time.time()
        ss_args['info']['solution_time'] = _direct_end-_direct_start

    if ss_args['return_info']:
        ss_args['info']['residual_norm'] = scipy.linalg.norm(b - L*v, np.inf)
        ss_args['info']['max_iter_refine'] = ss_args['max_iter_refine']
        ss_args['info']['scaling_vectors'] = ss_args['scaling_vectors']
        ss_args['info']['weighted_matching'] = ss_args['weighted_matching']

    if ss_args['use_rcm']:
        v = v[np.ix_(rev_perm,)]

    data = unstack_columns(_data.create(v), (n, n))
    data = _data.mul(_data.add(data, _data.adjoint(data)), 0.5)
    if ss_args['return_info']:
        return Qobj(data, dims=dims, isherm=True), ss_args['info']
    else:
        return Qobj(data, dims=dims, isherm=True)


def _steadystate_direct_dense(L, ss_args):
    """
    Direct solver that uses numpy arrays. Suitable for small systems with few
    states.
    """
    if settings.debug:
        logger.debug('Starting direct dense solver.')

    dims = L.dims[0]
    n = int(np.sqrt(L.shape[0]))
    b = np.zeros(n ** 2)
    b[0] = ss_args['weight']

    L = L.full()
    L[0, :] = np.diag(ss_args['weight'] * np.ones(n)).reshape(n ** 2)
    _dense_start = time.time()
    v = np.linalg.solve(L, b)
    _dense_end = time.time()
    ss_args['info']['solution_time'] = _dense_end-_dense_start
    if ss_args['return_info']:
        ss_args['info']['residual_norm'] = scipy.linalg.norm(b - L @ v, np.inf)
    data = unstack_columns(v)
    data = 0.5 * (data + data.conj().T)

    return Qobj(data, dims=dims, isherm=True)


def _steadystate_eigen(L, ss_args):
    """
    Internal function for solving the steady state problem by
    finding the eigenvector corresponding to the zero eigenvalue
    of the Liouvillian using ARPACK.
    """
    ss_args['info'].pop('weight', None)
    if settings.debug:
        logger.debug('Starting Eigen solver.')

    dims = L.dims[0]
    L = _data.to(_data.CSR, L.data).as_scipy().tocsc()

    if ss_args['use_rcm']:
        ss_args['info']['perm'].append('rcm')
        if settings.debug:
            old_band = _bandwidth(L)
            logger.debug('Original bandwidth: %i', old_band)
        perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L)
        rev_perm = np.argsort(perm)
        L = _permute(L, perm, perm)
        if settings.debug:
            rcm_band = _bandwidth(L)
            logger.debug('RCM bandwidth: %i', rcm_band)
            logger.debug('Bandwidth reduction factor: %f', old_band/rcm_band)

    _eigen_start = time.time()
    eigval, eigvec = eigs(L, k=1, sigma=1e-15, tol=ss_args['tol'],
                          which='LM', maxiter=ss_args['maxiter'])
    ss_args['info']['solution_time'] = time.time() - _eigen_start
    if ss_args['return_info']:
        ss_args['info']['residual_norm'] = scipy.linalg.norm(L*eigvec, np.inf)
    if ss_args['use_rcm']:
        eigvec = eigvec[np.ix_(rev_perm,)]
    size = int(np.sqrt(eigvec.size))
    data = _data.column_unstack(_data.dense.fast_from_numpy(eigvec), size)
    data = _data.mul(_data.add(data, _data.adjoint(data)), 0.5)
    out = Qobj(data, dims=dims, type='oper', isherm=True, copy=False)
    if ss_args['return_info']:
        return out/out.tr(), ss_args['info']
    else:
        return out/out.tr()


def _iterative_precondition(A, n, ss_args):
    """
    Internal function for preconditioning the steadystate problem for use
    with iterative solvers.
    """
    if settings.debug:
        logger.debug('Starting preconditioner.')
    _precond_start = time.time()
    try:
        P = spilu(A, permc_spec=ss_args['permc_spec'],
                  drop_tol=ss_args['drop_tol'],
                  diag_pivot_thresh=ss_args['diag_pivot_thresh'],
                  fill_factor=ss_args['fill_factor'],
                  options=dict(ILU_MILU=ss_args['ILU_MILU']))

        M = LinearOperator((n ** 2, n ** 2), matvec=P.solve)
        _precond_end = time.time()
        ss_args['info']['permc_spec'] = ss_args['permc_spec']
        ss_args['info']['drop_tol'] = ss_args['drop_tol']
        ss_args['info']['diag_pivot_thresh'] = ss_args['diag_pivot_thresh']
        ss_args['info']['fill_factor'] = ss_args['fill_factor']
        ss_args['info']['ILU_MILU'] = ss_args['ILU_MILU']
        ss_args['info']['precond_time'] = _precond_end-_precond_start

        if settings.debug or ss_args['return_info']:
            if settings.debug:
                logger.debug('Preconditioning succeeded.')
                logger.debug('Precond. time: %f' %
                             (_precond_end - _precond_start))
            L_nnz = P.L.nnz
            U_nnz = P.U.nnz
            ss_args['info']['l_nnz'] = L_nnz
            ss_args['info']['u_nnz'] = U_nnz
            ss_args['info']['ilu_fill_factor'] = (L_nnz+U_nnz)/A.nnz
            e = np.ones(n ** 2, dtype=int)
            condest = scipy.linalg.norm(M*e, np.inf)
            ss_args['info']['ilu_condest'] = condest
            if settings.debug:
                logger.debug('L NNZ: %i ; U NNZ: %i' % (L_nnz, U_nnz))
                logger.debug('Fill factor: %f' % ((L_nnz+U_nnz)/A.nnz))
                logger.debug('iLU condest: %f' % condest)

    except Exception:
        raise Exception("Failed to build preconditioner. Try increasing " +
                        "fill_factor and/or drop_tol.")

    return M, ss_args


def _steadystate_iterative(L, ss_args):
    """
    Iterative steady state solver using the GMRES, LGMRES, or BICGSTAB
    algorithm and a sparse incomplete LU preconditioner.
    """
    ss_iters = {'iter': 0}

    def _iter_count(r):
        ss_iters['iter'] += 1
        return

    if settings.debug:
        logger.debug('Starting %s solver.' % ss_args['method'])

    n = int(np.sqrt(L.shape[0]))
    dims = L.dims[0]
    b = np.zeros(n ** 2)
    b[0] = ss_args['weight']

    L, perm, perm2, rev_perm, ss_args = _steadystate_LU_liouvillian(L, ss_args)
    if np.any(perm):
        b = b[np.ix_(perm,)]
    if np.any(perm2):
        b = b[np.ix_(perm2,)]

    use_solver(assumeSortedIndices=True)

    if ss_args['M'] is None and ss_args['use_precond']:
        ss_args['M'], ss_args = _iterative_precondition(L, n, ss_args)
        if ss_args['M'] is None:
            warnings.warn("Preconditioning failed. Continuing without.",
                          UserWarning)

    # Select iterative solver type
    _iter_start = time.time()
    if ss_args['method'] == 'iterative-gmres':
        v, check = gmres(L, b, tol=ss_args['tol'], atol=ss_args['matol'],
                         M=ss_args['M'], x0=ss_args['x0'],
                         restart=ss_args['restart'],
                         maxiter=ss_args['maxiter'],
                         callback=_iter_count, callback_type='legacy')
    elif ss_args['method'] == 'iterative-lgmres':
        v, check = lgmres(L, b, tol=ss_args['tol'], atol=ss_args['matol'],
                          M=ss_args['M'], x0=ss_args['x0'],
                          maxiter=ss_args['maxiter'],
                          callback=_iter_count)
    elif ss_args['method'] == 'iterative-bicgstab':
        v, check = bicgstab(L, b, tol=ss_args['tol'],
                            atol=ss_args['matol'],
                            M=ss_args['M'], x0=ss_args['x0'],
                            maxiter=ss_args['maxiter'],
                            callback=_iter_count)
    else:
        raise Exception("Invalid iterative solver method.")
    _iter_end = time.time()

    ss_args['info']['iter_time'] = _iter_end - _iter_start
    if 'precond_time' in ss_args['info'].keys():
        ss_args['info']['solution_time'] = (ss_args['info']['iter_time'] +
                                            ss_args['info']['precond_time'])
    else:
        ss_args['info']['solution_time'] = ss_args['info']['iter_time']
    ss_args['info']['iterations'] = ss_iters['iter']
    if ss_args['return_info']:
        ss_args['info']['residual_norm'] = scipy.linalg.norm(b - L*v, np.inf)

    if settings.debug:
        logger.debug('Number of Iterations: %i', ss_iters['iter'])
        logger.debug('Iteration. time: %f', (_iter_end - _iter_start))

    if check > 0:
        raise Exception("Steadystate error: Did not reach tolerance after " +
                        str(ss_args['maxiter']) + " steps." +
                        "\nResidual norm: " +
                        str(ss_args['info']['residual_norm']))

    elif check < 0:
        raise Exception(
            "Steadystate error: Failed with fatal error: " + str(check) + ".")

    if ss_args['use_rcm']:
        v = v[np.ix_(rev_perm,)]

    size = int(np.sqrt(v.size))
    data = _data.column_unstack(_data.dense.fast_from_numpy(v), size)
    data = _data.mul(_data.add(data, _data.adjoint(data)), 0.5)
    if ss_args['return_info']:
        return Qobj(data, dims=dims, isherm=True), ss_args['info']
    else:
        return Qobj(data, dims=dims, isherm=True)


def _steadystate_svd_dense(L, ss_args):
    """
    Find the steady state(s) of an open quantum system by solving for the
    nullspace of the Liouvillian.
    """
    ss_args['info'].pop('weight', None)
    atol = 1e-12
    rtol = 1e-12
    if settings.debug:
        logger.debug('Starting SVD solver.')
    _svd_start = time.time()
    u, s, vh = np.linalg.svd(L.full(), full_matrices=False)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    _svd_end = time.time()
    ss_args['info']['solution_time'] = _svd_end-_svd_start
    if ss_args['all_states']:
        rhoss_list = []
        for n in range(ns.shape[1]):
            rhoss = Qobj(unstack_columns(ns[:, n]), dims=L.dims[0])
            rhoss_list.append(rhoss / rhoss.tr())
        if ss_args['return_info']:
            return rhoss_list, ss_args['info']
        else:
            if ss_args['return_info']:
                return rhoss_list, ss_args['info']
            else:
                return rhoss_list
    else:
        rhoss = Qobj(unstack_columns(ns[:, 0]), dims=L.dims[0])
        return rhoss / rhoss.tr()


def _steadystate_power_liouvillian(L, ss_args, has_mkl=0):
    """Creates modified Liouvillian for power based SS methods.
    """
    perm = None
    perm2 = None
    rev_perm = None
    n = L.shape[0]
    L = _data.sub(L.data, _data.identity(n, 1e-15))
    if ss_args['solver'] == 'mkl':
        kind = 'csr'
    else:
        kind = 'csc'
    if settings.debug:
        old_band = _bandwidth(L.as_scipy())
        old_pro = _profile(L.as_scipy())
        logger.debug('Original bandwidth: %i', old_band)
        logger.debug('Original profile: %i', old_pro)

    if ss_args['use_wbm']:
        if settings.debug:
            logger.debug('Calculating Weighted Bipartite Matching ordering...')
        _wbm_start = time.time()
        perm2 = _weighted_bipartite_matching(L.as_scipy())
        _wbm_end = time.time()
        L = _data.permute.indices_csr(L, perm, np.arange(n, dtype=np.int32))
        ss_args['info']['perm'].append('wbm')
        ss_args['info']['wbm_time'] = _wbm_end-_wbm_start
        if settings.debug:
            wbm_band = _bandwidth(L.as_scipy())
            wbm_pro = _profile(L.as_scipy())
            logger.debug('WBM bandwidth: %i', wbm_band)
            logger.debug('WBM profile: %i', wbm_pro)

    if ss_args['use_rcm']:
        if settings.debug:
            logger.debug('Calculating Reverse Cuthill-Mckee ordering...')
        ss_args['info']['perm'].append('rcm')
        _rcm_start = time.time()
        perm2 = scipy.sparse.csgraph.reverse_cuthill_mckee(L.as_scipy())
        _rcm_end = time.time()
        ss_args['info']['rcm_time'] = _rcm_end-_rcm_start
        rev_perm = np.argsort(perm2)
        L = _data.CSR(_permute(L.as_scipy(), perm2, perm2))
        if settings.debug:
            new_band = _bandwidth(L.as_scipy())
            new_pro = _profile(L.as_scipy())
            logger.debug('RCM bandwidth: %i', new_band)
            logger.debug('Bandwidth reduction factor: %f', old_band/new_band)
            logger.debug('RCM profile: %i', new_pro)
            logger.debug('Profile reduction factor: %f', old_pro/new_pro)
    L.sort_indices()
    L = L.as_scipy()
    if kind == 'csc':
        L = L.tocsc()
    return L, perm, perm2, rev_perm, ss_args


def _steadystate_power(L, ss_args):
    """
    Inverse power method for steady state solving.
    """
    ss_args['info'].pop('weight', None)
    if settings.debug:
        logger.debug('Starting iterative inverse-power method solver.')
    tol = ss_args['tol']
    mtol = ss_args['mtol']
    if mtol is None:
        mtol = max(0.1*tol, 1e-15)
    maxiter = ss_args['maxiter']

    use_solver(assumeSortedIndices=True)
    sflag = L.type == 'super'
    if sflag:
        rhoss_dims = L.dims[0]
    else:
        rhoss_dims = [L.dims[0], 1]
    n = L.shape[0]
    # Build Liouvillian
    if ss_args['solver'] == 'mkl' and ss_args['method'] == 'power':
        has_mkl = 1
    else:
        has_mkl = 0
    L, perm, perm2, rev_perm, ss_args = _steadystate_power_liouvillian(L,
                                                                       ss_args,
                                                                       has_mkl)
    orig_nnz = L.nnz
    # start with all ones as RHS
    v = np.ones(n, dtype=complex)
    if ss_args['use_rcm']:
        v = v[np.ix_(perm2,)]

    # Do preconditioning
    if ss_args['solver'] == 'scipy':
        if ss_args['M'] is None and ss_args['use_precond'] and \
                ss_args['method'] in ['power-gmres',
                                      'power-lgmres',
                                      'power-bicgstab']:
            ss_args['M'], ss_args = _iterative_precondition(L, int(np.sqrt(n)),
                                                            ss_args)
            if ss_args['M'] is None:
                warnings.warn("Preconditioning failed. Continuing without.",
                              UserWarning)

    ss_iters = {'iter': 0}

    def _iter_count(*args):
        ss_iters['iter'] += 1

    _power_start = time.time()
    # Get LU factors
    if ss_args['method'] == 'power':
        if ss_args['solver'] == 'mkl':
            lu = mkl_splu(L,
                          max_iter_refine=ss_args['max_iter_refine'],
                          scaling_vectors=ss_args['scaling_vectors'],
                          weighted_matching=ss_args['weighted_matching'])
        else:
            lu = splu(L,
                      permc_spec=ss_args['permc_spec'],
                      diag_pivot_thresh=ss_args['diag_pivot_thresh'],
                      options=dict(ILU_MILU=ss_args['ILU_MILU']))

            if settings.debug:
                L_nnz = lu.L.nnz
                U_nnz = lu.U.nnz
                logger.debug('L NNZ: %i ; U NNZ: %i', L_nnz, U_nnz)
                logger.debug('Fill factor: %f', (L_nnz+U_nnz)/orig_nnz)

    it = 0
    while (scipy.linalg.norm(L * v, np.inf) > tol) and (it < maxiter):
        check = 0
        if ss_args['method'] == 'power':
            v = lu.solve(v)
        elif ss_args['method'] == 'power-gmres':
            v, check = gmres(L, v, tol=mtol, atol=ss_args['matol'],
                             M=ss_args['M'], x0=ss_args['x0'],
                             restart=ss_args['restart'],
                             maxiter=ss_args['maxiter'],
                             callback=_iter_count,
                             callback_type='legacy')
        elif ss_args['method'] == 'power-lgmres':
            v, check = lgmres(L, v, tol=mtol, atol=ss_args['matol'],
                              M=ss_args['M'], x0=ss_args['x0'],
                              maxiter=ss_args['maxiter'],
                              callback=_iter_count)
        elif ss_args['method'] == 'power-bicgstab':
            v, check = bicgstab(L, v, tol=mtol, atol=ss_args['matol'],
                                M=ss_args['M'], x0=ss_args['x0'],
                                maxiter=ss_args['maxiter'],
                                callback=_iter_count)
        else:
            raise Exception("Invalid iterative solver method.")
        if check > 0:
            raise Exception("{} failed to find solution in "
                            "{} iterations.".format(ss_args['method'],
                                                    check))
        if check < 0:
            raise Exception("Breakdown in {}".format(ss_args['method']))
        v = v / scipy.linalg.norm(v, np.inf)
        it += 1
    if ss_args['method'] == 'power' and ss_args['solver'] == 'mkl':
        lu.delete()
        if ss_args['return_info']:
            ss_args['info']['max_iter_refine'] = ss_args['max_iter_refine']
            ss_args['info']['scaling_vectors'] = ss_args['scaling_vectors']
            ss_args['info']['weighted_matching'] = ss_args['weighted_matching']

    if it >= maxiter:
        raise Exception('Failed to find steady state after ' +
                        str(maxiter) + ' iterations')
    _power_end = time.time()
    ss_args['info']['solution_time'] = _power_end-_power_start
    ss_args['info']['iterations'] = it
    if ss_args['return_info']:
        ss_args['info']['residual_norm'] = scipy.linalg.norm(L*v, np.inf)
    if settings.debug:
        logger.debug('Number of iterations: %i', it)

    if ss_args['use_rcm']:
        v = v[np.ix_(rev_perm,)]

    # normalise according to type of problem
    if sflag:
        trow = v[::np.prod(rhoss_dims[0])+1]
        data = v / np.sum(trow)
    else:
        data = data / scipy.linalg.norm(v)

    data = unstack_columns(_data.create(data))
    data = _data.mul(_data.add(data, _data.adjoint(data)), 0.5)
    rhoss = Qobj(data,
                 dims=rhoss_dims,
                 isherm=True,
                 copy=False)
    if ss_args['return_info']:
        return rhoss, ss_args['info']
    else:
        return rhoss


def steadystate_floquet(H_0, c_ops, Op_t, w_d=1.0, n_it=3, sparse=False):
    """
    Calculates the effective steady state for a driven
     system with a time-dependent cosinusoidal term:
    .. math::
        \\mathcal{\\hat{H}}(t) = \\hat{H}_0 +
         \\mathcal{\\hat{O}} \\cos(\\omega_d t)
    Parameters
    ----------
    H_0 : :obj:`~Qobj`
        A Hamiltonian or Liouvillian operator.
    c_ops : list
        A list of collapse operators.
    Op_t : :obj:`~Qobj`
        The the interaction operator which is multiplied by the cosine
    w_d : float, default 1.0
        The frequency of the drive
    n_it : int, default 3
        The number of iterations for the solver
    sparse : bool, default False
        Solve for the steady state using sparse algorithms.
        Actually, dense seems to be faster.
    Returns
    -------
    dm : qobj
        Steady state density matrix.
    .. note::
        See: Sze Meng Tan,
        https://copilot.caltech.edu/documents/16743/qousersguide.pdf,
        Section (10.16)
    """
    if False:
        # TODO: rewrite using `core.Data`
        N = H_0.shape[0]

        L_0 = liouvillian(H_0, c_ops).data.tocsc()
        L_t = liouvillian(Op_t)
        L_p = (0.5 * L_t).data.tocsc()
        # L_p and L_m correspond to the positive and negative
        # frequency terms respectively.
        # They are independent in the model, so we keep both names.
        L_m = L_p
        L_p_array = L_p.todense()
        L_m_array = L_p_array

        Id = scipy.sparse.eye(N ** 2, format="csc", dtype=np.complex128)
        S = T = scipy.sparse.csc_matrix((N ** 2, N ** 2), dtype=np.complex128)

        for n_i in np.arange(n_it, 0, -1):
            L = scipy.sparse.csc_matrix(L_0 - 1j * n_i * w_d * Id + L_m.dot(S))
            L.sort_indices()
            LU = splu(L)
            S = - LU.solve(L_p_array)

            L = scipy.sparse.csc_matrix(L_0 + 1j * n_i * w_d * Id + L_p.dot(T))
            L.sort_indices()
            LU = splu(L)
            T = - LU.solve(L_m_array)

        M_subs = L_0 + L_m.dot(S) + L_p.dot(T)
    else:
        N = H_0.shape[0]

        L_0 = liouvillian(H_0, c_ops).full()
        L_t = liouvillian(Op_t)
        L_p = (0.5 * L_t).full()
        L_m = L_p

        Id = np.eye(N ** 2)
        S, T = np.zeros((N ** 2, N ** 2)), np.zeros((N ** 2, N ** 2))

        for n_i in np.arange(n_it, 0, -1):
            L = L_0 - 1j * n_i * w_d * Id + np.matmul(L_m, S)
            lu, piv = scipy.linalg.lu_factor(L)
            S = - scipy.linalg.lu_solve((lu, piv), L_p)

            L = L_0 + 1j * n_i * w_d * Id + np.matmul(L_p, T)
            lu, piv = scipy.linalg.lu_factor(L)
            T = - scipy.linalg.lu_solve((lu, piv), L_m)

        M_subs = L_0 + np.matmul(L_m, S) + np.matmul(L_p, T)

    return steadystate(Qobj(M_subs, type="super", dims=L_t.dims))


def build_preconditioner(A, c_op_list=[], **kwargs):
    """Constructs a iLU preconditioner necessary for solving for
    the steady state density matrix using the iterative linear solvers
    in the 'steadystate' function.

    Parameters
    ----------
    A : qobj
        A Hamiltonian or Liouvillian operator.

    c_op_list : list
        A list of collapse operators.

    return_info : bool, optional, default = False
        Return a dictionary of solver-specific infomation about the
        solution and how it was obtained.

    use_rcm : bool, optional, default = False
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the
        LU factorization of the Liouvillian.

    use_wbm : bool, optional, default = False
        Use Weighted Bipartite Matching reordering to make the Liouvillian
        diagonally dominant.  This is useful for iterative preconditioners
        only, and is set to ``True`` by default when finding a preconditioner.

    weight : float, optional
        Sets the size of the elements used for adding the unity trace condition
        to the linear solvers.  This is set to the average abs value of the
        Liouvillian elements if not specified by the user.

    method : str, default = 'iterative'
        Tells the preconditioner what type of Liouvillian to build for
        iLU factorization.  For direct iterative methods use 'iterative'.
        For power iterative methods use 'power'.

    permc_spec : str, optional, default='COLAMD'
        Column ordering used internally by superLU for the
        'direct' LU decomposition method. Options include 'COLAMD' and
        'NATURAL'. If using RCM then this is set to 'NATURAL' automatically
        unless explicitly specified.

    fill_factor : float, optional, default = 100
        Specifies the fill ratio upper bound (>=1) of the iLU
        preconditioner.  Lower values save memory at the cost of longer
        execution times and a possible singular factorization.

    drop_tol : float, optional, default = 1e-4
        Sets the threshold for the magnitude of preconditioner
        elements that should be dropped.  Can be reduced for a courser
        factorization at the cost of an increased number of iterations, and a
        possible singular factorization.

    diag_pivot_thresh : float, optional, default = None
        Sets the threshold between [0,1] for which diagonal
        elements are considered acceptable pivot points when using a
        preconditioner.  A value of zero forces the pivot to be the diagonal
        element.

    ILU_MILU : str, optional, default = 'smilu_2'
        Selects the incomplete LU decomposition method algoithm used in
        creating the preconditoner. Should only be used by advanced users.

    Returns
    -------
    lu : object
        Returns a SuperLU object representing iLU preconditioner.

    info : dict, optional
        Dictionary containing solver-specific information.
    """
    ss_args = _default_steadystate_args()
    ss_args['method'] = 'iterative'
    for key, value in kwargs.items():
        if key in ss_args:
            ss_args[key] = value
        else:
            raise TypeError("Invalid keyword argument '" + key +
                            "' passed to steadystate.")

    # Set column perm to NATURAL if using RCM and not specified by user
    if ss_args['use_rcm'] and ('permc_spec' not in kwargs):
        ss_args['permc_spec'] = 'NATURAL'

    L = _steadystate_setup(A, c_op_list)
    # Set weight parameter to max abs val in L if not set explicitly
    if 'weight' not in kwargs.keys():
        ss_args['weight'] = np.abs(_data.norm.max(L.data))
        ss_args['info']['weight'] = ss_args['weight']

    n = int(np.sqrt(L.shape[0]))
    if ss_args['method'] == 'iterative':
        ss_list = _steadystate_LU_liouvillian(L, ss_args)
        L, _, _, _, ss_args = ss_list
    elif ss_args['method'] == 'power':
        ss_list = _steadystate_power_liouvillian(L, ss_args)
        L, _, _, _, ss_args = ss_list
    else:
        raise ValueError("Invalid preconditioning method.")

    M, ss_args = _iterative_precondition(L, n, ss_args)

    if ss_args['return_info']:
        return M, ss_args['info']
    else:
        return M


def _pseudo_inverse_dense(L, rhoss, w=None, **pseudo_args):
    """
    Internal function for computing the pseudo inverse of an Liouvillian using
    dense matrix methods. See pseudo_inverse for details.
    """
    rho_vec = np.transpose(stack_columns(rhoss.full()))

    tr_mat = tensor([identity(n) for n in L.dims[0][0]])
    tr_vec = np.transpose(stack_columns(tr_mat.full()))
    N = np.prod(L.dims[0][0])
    I = np.identity(N * N)
    P = np.kron(np.transpose(rho_vec), tr_vec)
    Q = I - P

    if w is None:
        L = L
    else:
        L = 1.0j*w*spre(tr_mat)+L

    if pseudo_args['method'] == 'direct':
        try:
            LIQ = np.linalg.solve(L.full(), Q)
        except Exception:
            LIQ = np.linalg.lstsq(L.full(), Q)[0]

        R = np.dot(Q, LIQ)

        return Qobj(R, dims=L.dims)

    elif pseudo_args['method'] == 'numpy':
        return Qobj(np.dot(Q, np.dot(np.linalg.pinv(L.full()), Q)),
                    dims=L.dims)

    elif pseudo_args['method'] == 'scipy':
        # return Qobj(la.pinv(L.full()), dims=L.dims)
        return Qobj(np.dot(Q, np.dot(scipy.linalg.pinv(L.full()), Q)),
                    dims=L.dims)

    elif pseudo_args['method'] == 'scipy2':
        # return Qobj(la.pinv2(L.full()), dims=L.dims)
        return Qobj(np.dot(Q, np.dot(scipy.linalg.pinv2(L.full()), Q)),
                    dims=L.dims)

    else:
        raise ValueError(
            "Unsupported method '%s'. Use 'direct', 'numpy', 'scipy' or"
            " 'scipy2'" % pseudo_args['method'])


def _pseudo_inverse_sparse(L, rhoss, w=None, **pseudo_args):
    """
    Internal function for computing the pseudo inverse of an Liouvillian using
    sparse matrix methods. See pseudo_inverse for details.
    """

    L = L.to(_data.CSR)
    N = np.prod(L.dims[0][0])

    rhoss_vec = operator_to_vector(rhoss)

    tr_op = tensor([identity(n) for n in L.dims[0][0]])
    tr_op_vec = operator_to_vector(tr_op)

    P = _data.kron(rhoss_vec.data, tr_op_vec.data.transpose(), dtype=_data.CSR)
    I = _data.csr.identity(N * N)
    Q = _data.sub(I, P)

    if w is None:
        L = 1e-15j*spre(tr_op) + L
    else:
        if w != 0.0:
            L = 1.0j*w*spre(tr_op) + L
        else:
            L = 1e-15j*spre(tr_op) + L

    if pseudo_args['use_rcm']:
        perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L.data.as_scipy())
        A = _data.permute.indices(L.data, perm, perm, dtype=_data.CSR)
        Q = _data.permute.indices(Q, perm, perm, dtype=_data.CSR)
    else:
        if pseudo_args['solver'] == 'scipy':
            A = L.data.as_scipy().tocsc()
            A.sort_indices()

    if pseudo_args['method'] == 'splu':
        if settings.has_mkl:
            L.data.sort_indices()
            A = L.data.as_scipy()
            LIQ = mkl_spsolve(A, Q.to_array())
        else:
            pspec = pseudo_args['permc_spec']
            diag_p_thresh = pseudo_args['diag_pivot_thresh']
            lu = splu(A,
                      permc_spec=pspec,
                      diag_pivot_thresh=diag_p_thresh,
                      options={'ILU_MILU': pseudo_args['ILU_MILU']})
            LIQ = lu.solve(Q.to_array())

    elif pseudo_args['method'] == 'spilu':
        lu = spilu(A,
                   permc_spec=pseudo_args['permc_spec'],
                   fill_factor=pseudo_args['fill_factor'],
                   drop_tol=pseudo_args['drop_tol'])
        LIQ = lu.solve(Q.to_array())

    else:
        raise ValueError("unsupported method '%s'" % pseudo_args['method'])

    R = _data.matmul(Q, _data.create(LIQ))

    if pseudo_args['use_rcm']:
        rev_perm = np.argsort(perm)
        R = _data.permute.indices(R, rev_perm, rev_perm)

    return Qobj(R, dims=L.dims)


def pseudo_inverse(L, rhoss=None, w=None, sparse=True,
                   method='splu', **kwargs):
    """
    Compute the pseudo inverse for a Liouvillian superoperator, optionally
    given its steady state density matrix (which will be computed if not
    given).

    Returns
    -------
    L : Qobj
        A Liouvillian superoperator for which to compute the pseudo inverse.


    rhoss : Qobj
        A steadystate density matrix as Qobj instance, for the Liouvillian
        superoperator L.

    w : double
        frequency at which to evaluate pseudo-inverse.  Can be zero for dense
        systems and large sparse systems. Small sparse systems can fail for
        zero frequencies.

    sparse : bool
        Flag that indicate whether to use sparse or dense matrix methods when
        computing the pseudo inverse.

    method : string
        Name of method to use. For sparse=True, allowed values are 'spsolve',
        'splu' and 'spilu'. For sparse=False, allowed values are 'direct' and
        'numpy'.

    kwargs : dictionary
        Additional keyword arguments for setting parameters for solver methods.

    Returns
    -------
    R : Qobj
        Returns a Qobj instance representing the pseudo inverse of L.

    Note
    ----
    In general the inverse of a sparse matrix will be dense.  If you
    are applying the inverse to a density matrix then it is better to
    cast the problem as an Ax=b type problem where the explicit calculation
    of the inverse is not required. See page 67 of "Electrons in
    nanostructures" C. Flindt, PhD Thesis available online:
    https://orbit.dtu.dk/fedora/objects/orbit:82314/datastreams/
    file_4732600/content

    Note also that the definition of the pseudo-inverse herein is different
    from numpys pinv() alone, as it includes pre and post projection onto
    the subspace defined by the projector Q.

    """
    pseudo_args = _default_steadystate_args()
    for key, value in kwargs.items():
        if key in pseudo_args:
            pseudo_args[key] = value
        else:
            raise TypeError(
                "Invalid keyword argument '"+key+"' passed to pseudo_inverse.")
    pseudo_args['method'] = method

    # Set column perm to NATURAL if using RCM and not specified by user
    if pseudo_args['use_rcm'] and ('permc_spec' not in kwargs):
        pseudo_args['permc_spec'] = 'NATURAL'

    if rhoss is None:
        rhoss = steadystate(L, **pseudo_args)

    if sparse:
        return _pseudo_inverse_sparse(L, rhoss, w=w, **pseudo_args)
    else:
        if pseudo_args['method'] != 'splu':
            pseudo_args['method'] = pseudo_args['method']
        else:
            pseudo_args['method'] = 'direct'
        return _pseudo_inverse_dense(L, rhoss, w=w, **pseudo_args)
