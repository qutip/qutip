from qutip import liouvillian, lindblad_dissipator, Qobj
import qutip.core.data as _data


def _permute_wbm(L, b):
    perm = scipy.sparse.csgraph.maximum_bipartite_matching(L.as_scipy())
    L = _data.permute.indices(L, perm, None)
    b = _data.permute.indices(b, perm, None)
    return L, b


def _permute_rcm(L, b):
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L.as_scipy())
    L = _data.permute.indices(L, perm, perm)
    b = _data.permute.indices(b, perm, None)
    return L, b, perm


def _reverse_rcm(rho, perm):
    rev_perm = np.argsort(perm)
    rho = _data.permute.indices(rho, rev_perm, None)
    return rho


def steadystate(A, c_ops=[], *, method='direct', solve_method=None, weight=0, 
                **kwargs):
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

    method : str, default='direct'
        The allowed methods are composed of 2 parts, the steadystate method:
        - 'direct' or 'iterative': Solving ``L(rho_ss) = 0``
        - 'eigen' : Eigenvalue problem
        - 'svd' : Singular value decomposition
        - 'power': Inverse-power method

    solver : str, default=None
        'direct' and 'power' methods only.
        Solver to use when solving the ``L(rho) = 0`` equation.
        Default supported solver are:
        - "solve", "lstsq":
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab":
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve",
          sparse solver by mkl.
        Extension to qutip, such as qutip-tensorflow, can use come with their
        own solver. When ``A`` and ``c_ops`` use these data backends, see the
        corresponding libraries ``linalg`` for available solver.

        Extra options for these solver can be passed in ``**kw``.

    **kw :
        use_rcm
        use_wbm
        weigth
        sparse
        tol, max_iter


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
    if not A.issuper and not c_ops:
        raise TypeError('Cannot calculate the steady state for a ' +
                        'non-dissipative system.')
    if not A.issuper:
        A = liouvillian(A)
    for op in c_ops:
        A += lindblad_dissipator(op)

    if "-" in method:
        method, solver = method.split("-")

    # We want the user to be able to use this without having to know what data
    # type the liouvillian use. For extra data types (tensorflow) we can expect
    # the users to know they are using them and choose an appropriate solver
    sparse_solvers = ["spsolve", "mkl_spsolve", "gmres", "lgmres", "bicgstab"]
    if isinstance(A.data, _data.csr) and solver in ["solve", "lstsq"]:
        A = A.to("dense")
    elif isinstance(A.data, _data.Dense) and solver in sparse_solvers:
        A = A.to("csr")

    if method in ["direct", "iterative"]:
        return _steadystate_direct(A, weight, method=solver, **kwargs)
    elif method == "eigen":
        return _steadystate_eigen(A, **kwargs)
    elif method == "svd":
        return _steadystate_svd(A, **kwargs)
    elif method == "power":
        return _steadystate_power(A, method=solver, **kwargs)
    else:
        raise ValueError(f"method {method} not supported.")


def _steadystate_direct(A, weight, **kw):
    # Find the weight, no good dispatched function available...
    if weight:
        pass
    elif isinstance(A.data, _data.CSR):
        weight = np.mean(np.abs(A.data.as_scipy().data))
    else:
        A_np = np.abs(A.full())
        weight = np.mean(A_np[A_np > 0])

    # Add weight to the Liouvillian
    # A[:, 0] = vectorized(eye * weight)
    # We don't have a function to overwrite part of an array, so
    N = A.shape[0]
    n = int(A.shape[0]**0.5)
    dtype = type(A.data)
    L_row0 = _data.matmul(_data.one_element[dtype]((1, N), (0, 0), 1), A.data)
    weight_vec = _data.column_stack(_data.diag([weight] * n, 0, dtype=dtype))
    weight_vec = _data.add(weight_vec.transpose(), L_row0, -1)
    weight_mat = _data.kron(
        weight_vec,
        _data.one_element[dtype]((N, 1), (0, 0), 1)
    )
    L = _data.add(weight_mat, A.data)
    b = _data.one_element[dtype]((N, 1), (0, 0), weight)

    # Permutation are part of scipy.sparse, thus only supported for CSR.
    if kw.pop("use_wbm", False) and isinstance(L, _data.CSR):
        L, b = _permute_wbm(L, b)
    use_rcm = kw.pop("use_rcm", False) and isinstance(L, _data.CSR)
    if use_rcm:
        L, b, perm = _permute_rcm(L, b)

    steadystate = _data.solve(L, b, **kw)

    if use_rcm:
        steadystate = _reverse_rcm(steadystate, perm)

    rho_ss = _data.column_unstack(steadystate, n)
    rho_ss = _data.add(rho_ss, rho_ss.adjoint()) * 0.5

    return Qobj(rho_ss, dims=A.dims[0], isherm=True)


def _steadystate_eigen(L, **kw):
    # v4's implementation only uses sparse eigen solver
    val, vec = (L.dag() @ L).eigenstates(
        eigvals=1,
        sort="low",
        sparse=kw.pop("sparse", True)
    )
    rho = qt.vector_to_operator(vec[0])
    return rho / rho.tr()


def _steadystate_svd(L, **kw):
    u, s, vh = _data.svd(L.data, True)
    vec = Qobj(_data.split_columns(vh.adjoint())[-1], dims=[L.dims[0],[1]])
    rho = qt.vector_to_operator(vec)
    return rho / rho.tr()


def _steadystate_power(A, **kw):
    A += 1e-15
    L = A.data
    N = L.shape[1]
    y = _data.Dense([1]*N)

    # Permutation are part of scipy.sparse, thus only supported for CSR.
    if kw.pop("use_wbm", False) and isinstance(L, _data.CSR):
        L, y = _permute_wbm(L, y)
    use_rcm = kw.pop("use_rcm", False) and isinstance(L, _data.CSR)
    if use_rcm:
        L, y, perm = _permute_rcm(L, y)

    it = 0
    maxiter = kw.pop("maxiter", 1000)
    tol = kw.pop("tol", 1e-12)
    while it < maxiter and _data.norm.max(L @ y) > tol:
        y = _data.solve(L, y, **kw)
        y = y / _data.norm.max(y)
        it += 1

    if it >= maxiter:
        raise Exception('Failed to find steady state after ' +
                        str(maxiter) + ' iterations')

    if use_rcm:
        y = _reverse_rcm(y, perm)

    rho_ss = Qobj(_data.column_unstack(y, N**0.5), dims=A.dims[0], isherm=True)
    return rho_ss / rho_ss.tr()
