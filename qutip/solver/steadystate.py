from qutip import liouvillian, lindblad_dissipator, Qobj
import qutip.data as _data

def steadystate(A, c_ops=[], *, method='direct', weight=0, **kwargs):
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

    method : str, default 'direct'
        The allowed methods are
        - 'direct'
        - 'eigen'
        - 'svd'
        - 'power'

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
        A += lindbald_dissipator(op)

    if method == "direct":
        return _steadystate_direct(A, weight, **kwargs)
    elif method in ["eigen", "svn"]:
        return _steadystate_decomposition(A, method, **kwargs)
    elif method == "power":
        return _steadystate_power(A, **kwargs)
    else:
        raise ValueError(f"method {mehtod} not supported.")

def _steadystate_direct(A, weight, **kw):
    # Find Add weight
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
    weight_mat = _data.kron(weight_vec, _data.one_element[dtype]((N, 1), (0, 0), 1))
    A += Qobj(weight_mat, dims=A.dims)

    b = _data.one_element[dtype]((N, 1), (0, 0), weight)

    steadystate = _data.solve(A.data, b, **kw)
    rho_ss = _data.column_unstack(steadystate, n)
    rho_ss = _data.add(rho_ss, rho_ss.adjoint()) * 0.5

    return Qobj(rho_ss, dims=A.dims[0], isherm=True)

def _steadystate_decomposition(L, method="eigen", **kw):
    if method == "eigen":
        val, vec = (L.dag() @ L).eigenstates(eigvals=1, sort="low", sparse=kw.pop("sparse", True))
        vec = vec[0]

    elif method == "svd":
        u, s, vh = _data.svd(L.data, True)
        vec = Qobj(_data.split_columns(vh.adjoint())[-1], dims=L.dims[0])

    return rho / rho.tr()

def _steadystate_power(L, **kw):
    L += 1e-15
    N = L.shape[1]
    y = _data.Dense([1]*N)
    it = 0
    tol = 1e-10
    while it < 5 and _data.norm.max(L.data @ y) > tol:
        y = _data.solve(L.data, y, **kw)
        y = y / _data.norm.max(y)

    rho_ss = Qobj(_data.column_unstack(y, n), dims=A.dims[0], isherm=True)
    return rho_ss / rho_ss.tr()
