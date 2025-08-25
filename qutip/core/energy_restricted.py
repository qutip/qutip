from .dimensions import Space
from .states import state_number_enumerate
from . import data as _data
from . import Qobj, qdiags
import numpy as np
import scipy.sparse
from .. import settings
import math
import numbers
import itertools
import warnings

__all__ = ['enr_state_dictionaries', 'enr_nstates',
           'enr_fock', 'enr_thermal_dm', 'enr_destroy', 'enr_identity',
           'enr_ptrace', 'enr_tensor']


def enr_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.

    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.

    excitations : integer
        The maximum numbers of dimension

    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices. state2idx and idx2state are reverses
        of each other, i.e., ``state2idx[idx2state[idx]] = idx`` and
        ``idx2state[state2idx[state]] = state``.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state


def enr_nstates(dims, excitations):
    """
    Directly compute the number of states for a system with a given number of
    components and maximum number of excitations, using the inclusion-exclusion
    principle. Much faster than enumerating all states.

    Parameters
    ----------
    dims: list of integers
        A list with the number of states in each sub-system.

    excitations : integer
        The maximum number excitations across all sub-systems.

    Returns
    -------
    nstates: integer
        The number of states in the excitation-number restricted state space
    """
    if len(dims) == 0:
        return 1
    m = len(dims)
    kmax = min((excitations, sum(dims)))//min(dims)
    if all(d == dims[0] for d in dims):  # this situation can be solved faster
        return sum(
            (-1)**k * math.comb(m, k) * math.comb(excitations-k*dims[0]+m, m)
            for k in range(kmax+1))
    else:  # in general, need to iterate over all subsets
        return sum((-1)**k * math.comb(excitations + m - sum(subset), m)
                   for k in range(kmax+1)
                   for subset in itertools.combinations(dims, k)
                   if sum(subset) < excitations+m)


class EnrSpace(Space):
    _stored_dims = {}

    def __init__(self, dims, excitations):
        self.dims = tuple(dims)
        self.n_excitations = excitations
        enr_dicts = enr_state_dictionaries(dims, excitations)
        self.size, self.state2idx, self.idx2state = enr_dicts
        self.issuper = False
        self.superrep = None
        self._pure_dims = False

    def __eq__(self, other):
        return (
            self is other
            or (
                type(other) is type(self)
                and self.dims == other.dims
                and self.n_excitations == other.n_excitations
            )
        )

    def __hash__(self):
        return hash((self.dims, self.n_excitations))

    def __repr__(self):
        return f"EnrSpace({self.dims}, {self.n_excitations})"

    def as_list(self):
        return list(self.dims)

    def dims2idx(self, dims):
        return self.state2idx[tuple(dims)]

    def idx2dims(self, idx):
        return self.idx2state[idx]


def enr_fock(dims, excitations, state, *, dtype=None):
    """
    Generate the Fock state representation in a excitation-number restricted
    state space. The `dims` argument is a list of integers that define the
    number of quantums states of each component of a composite quantum system,
    and the `excitations` specifies the maximum number of excitations for
    the basis states that are to be included in the state space. The `state`
    argument is a tuple of integers that specifies the state (in the number
    basis representation) for which to generate the Fock state representation.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    state : list of integers
        The state in the number basis representation.

    dtype : type or str, optional
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    ket : Qobj
        A Qobj instance that represent a Fock state in the exication-number-
        restricted state space defined by `dims` and `exciations`.

    """
    dtype = _data._parse_default_dtype(dtype, "dense")
    nstates, state2idx, _ = enr_state_dictionaries(dims, excitations)
    try:
        data = _data.one_element[dtype](
            (nstates, 1),
            (state2idx[tuple(state)], 0),
            1
        )
    except KeyError:
        msg = (
            "state tuple " + str(tuple(state))
            + " is not in the restricted state space."
        )
        raise ValueError(msg) from None
    return Qobj(data, dims=[EnrSpace(dims, excitations), [1]*len(dims)],
                copy=False, dtype=dtype)


def enr_thermal_dm(dims, excitations, n, *, dtype=None):
    """
    Generate the density operator for a thermal state in the excitation-number-
    restricted state space defined by the `dims` and `exciations` arguments.
    See the documentation for enr_fock for a more detailed description of
    these arguments. The temperature of each mode in dims is specified by
    the average number of excitatons `n`.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    n : integer
        The average number of exciations in the thermal state. `n` can be
        a float (which then applies to each mode), or a list/array of the same
        length as dims, in which each element corresponds specifies the
        temperature of the corresponding mode.

    dtype : type or str, optional
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    dm : Qobj
        Thermal state density matrix.
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    nstates, _, idx2state = enr_state_dictionaries(dims, excitations)
    enr_dims = [EnrSpace(dims, excitations)] * 2
    if not isinstance(n, (list, np.ndarray)):
        n = np.ones(len(dims)) * n
    else:
        n = np.asarray(n)

    diags = [np.prod((n / (n + 1)) ** np.array(state))
             for idx, state in idx2state.items()]
    diags /= np.sum(diags)
    out = qdiags(diags, 0, dims=enr_dims,
                 shape=(nstates, nstates), dtype=dtype)
    out._isherm = True
    return out


def enr_destroy(dims, excitations, *, dtype=None):
    """
    Generate annilation operators for modes in a excitation-number-restricted
    state space. For example, consider a system consisting of 4 modes, each
    with 5 states. The total hilbert space size is 5**4 = 625. If we are
    only interested in states that contain up to 2 excitations, we only need
    to include states such as

        (0, 0, 0, 0)
        (0, 0, 0, 1)
        (0, 0, 0, 2)
        (0, 0, 1, 0)
        (0, 0, 1, 1)
        (0, 0, 2, 0)
        ...

    This function creates annihilation operators for the 4 modes that act
    within this state space:

        a1, a2, a3, a4 = enr_destroy([5, 5, 5, 5], excitations=2)

    From this point onwards, the annihiltion operators a1, ..., a4 can be
    used to setup a Hamiltonian, collapse operators and expectation-value
    operators, etc., following the usual pattern.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    dtype : type or str, optional
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    a_ops : list of qobj
        A list of annihilation operators for each mode in the composite
        quantum system described by dims.
    """
    dtype = _data._parse_default_dtype(dtype, "sparse")
    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)
    enr_dims = [EnrSpace(dims, excitations)] * 2

    a_ops = [scipy.sparse.lil_matrix((nstates, nstates), dtype=np.complex128)
             for _ in dims]

    for n1, state1 in idx2state.items():
        for idx, s in enumerate(state1):
            # if s > 0, the annihilation operator of mode idx has a non-zero
            # entry with one less excitation in mode idx in the final state
            if s > 0:
                state2 = state1[:idx] + (s-1,) + state1[idx+1:]
                n2 = state2idx[state2]
                a_ops[idx][n2, n1] = np.sqrt(s)

    return [
        Qobj(a, dims=enr_dims, isunitary=False, isherm=False, dtype=dtype)
        for a in a_ops
    ]


def enr_identity(dims, excitations, *, dtype=None):
    """
    Generate the identity operator for the excitation-number restricted
    state space defined by the `dims` and `exciations` arguments. See the
    docstring for enr_fock for a more detailed description of these arguments.

    Parameters
    ----------
    dims : list
        A list of the dimensions of each subsystem of a composite quantum
        system.

    excitations : integer
        The maximum number of excitations that are to be included in the
        state space.

    dtype : type or str, optional
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    op : Qobj
        A Qobj instance that represent the identity operator in the
        exication-number-restricted state space defined by `dims` and
        `exciations`.
    """
    dtype = _data._parse_default_dtype(dtype, "diagonal")
    dims = EnrSpace(dims, excitations)
    return Qobj(_data.identity[dtype](dims.size),
                dims=[dims, dims],
                isherm=True,
                isunitary=True,
                copy=False,
                dtype=dtype)


def enr_ptrace(rho, sel):
    """
    Trace out the modes not in `sel`, with input and output in the excitation-
    number restricted state space. Only for density matrices with CSR dtype.
    Be aware that the partial trace behaves weirdly in ENR space, since the
    size of the subsystems' spaces depend on other subsystems' states. e.g.,
    it does not invert the tensor multiplication with the identity operator.

    Parameters
    ----------
    rho : Qobj
        The input density matrix.

    sel : list of integers
        The indices of the modes to keep.

    Returns
    -------
    out : Qobj
        A Qobj instance that represents the partially-traced `rho` in the
        excitation-number restricted state. The maximum number of
        excitations is the same as for the input.
    """
    if rho.shape[0] != rho.shape[1]:
        raise ValueError(
            "enr_ptrace is only defined for square density matrices")
    try:
        sel = sorted(sel)
    except TypeError:
        if not isinstance(sel, numbers.Integral):
            raise TypeError(
                "selection must be an integer or list of integers"
            ) from None
        sel = [sel]

    dims = rho.dims[0]
    if (len(sel) == len(dims)):
        # no need to trace out anything
        return rho
    if len(sel) == 0:
        # trace out all modes
        return Qobj(rho.tr(), dtype="csr")

    dtype = rho.dtype
    if dtype is _data.Dense:
        warnings.warn("enr_ptrace may be slow for dense matrices.")

    excitations = rho._dims[0].n_excitations
    mat = rho.to(_data.CSR).data.as_scipy().tocoo()

    # get the new dimensions
    dims_new = [dims[i] for i in sel]
    toremove = set(range(len(dims))) - set(sel)
    # initialize the new matrix
    space_new = EnrSpace(dims_new, excitations)
    out = scipy.sparse.dok_matrix((space_new.size, space_new.size),
                                  dtype="complex128")

    # loop over nonzero elements of the input, (it's sparse)
    for row_idx_old, col_idx_old, val in zip(mat.row, mat.col, mat.data):
        # find the states corresponding to the old indices
        row_state_old = rho._dims[0].idx2dims(row_idx_old)
        col_state_old = rho._dims[0].idx2dims(col_idx_old)

        # only keep if on diagonal or if diagonal wrt systems to be removed
        if (row_idx_old == col_idx_old or all(
                row_state_old[ii] == col_state_old[ii] for ii in toremove)):
            # find the new indices for the kept subsystems
            row_state_new = tuple(row_state_old[ii] for ii in sel)
            col_state_new = tuple(col_state_old[ii] for ii in sel)

            row_idx_new = space_new.dims2idx(row_state_new)
            col_idx_new = space_new.dims2idx(col_state_new)
            out[row_idx_new, col_idx_new] += val

    # convert to Qobj
    return Qobj(out, dims=[space_new]*2, dtype=dtype)


def enr_tensor(*args: Qobj, newexcitations: int = None,
               truncate=False, verbose=True):
    """
    Calculates the tensor product of input operators in the excitation-
    number restricted state space.
    Be aware that tensor multiplication behaves weirdly in ENR space, since the
    size of the subsystems' spaces depend on other subsystems' states.
    Taking a tensor product with the identity is not inverted by partial trace!

    Parameters
    ----------
    args : Qobj's
        The input operators to be tensor-multiplied.

    newexcitations : integer, optional
        The maximum number of excitations for the output operator. If not
        given, the minimum of the input excitations is used.

    truncate : bool, optional
        If True, the function will ignore entries that are not in the new
        restricted state space. If False, such entries will raise an error.

    verbose : bool, optional
        If True, the function will print a message if any entries are
        truncated.

    Returns
    -------
    obj : qobj
        A composite quantum object.
    """
    if not args:
        raise TypeError("Requires at least one input argument")
    if len(args) == 1 and isinstance(args[0], Qobj):
        return args[0].copy()
    if len(args) == 1:
        try:
            args = tuple(args[0])
        except TypeError:
            raise TypeError("requires Qobj operands") from None
    if not all(isinstance(q, Qobj) for q in args):
        raise TypeError("requires Qobj operands")

    excitations = [(getattr(q._dims[0], 'n_excitations', None)
                    or q._dims[1].n_excitations) for q in args]

    # get the new max excitations
    newexcitations = newexcitations or min(excitations)
    newdims = [tuple(itertools.chain(*(q.dims[i] for q in args)))
               for i in range(2)]

    trunccount = 0
    out = {((), ()): 1.0}  # initialize a dict to tensor the first Qobj with

    # loop over the the operators
    for q in args:
        out, trunccount = _enr_tensor_qobj_with_dict(
            out, q.to(_data.CSR), newexcitations, truncate, trunccount)

    if trunccount > 0 and verbose:
        print(f"Truncated {trunccount} entries.")

    dtype = args[0].dtype if all(q.dtype == args[0].dtype for q in args) \
        else None
    dtype = _data._parse_default_dtype(dtype, "sparse")
    return _enr_qobj_from_dict(out, newdims, newexcitations,
                               isherm=all(q.isherm for q in args), dtype=dtype)


def _enr_tensor_qobj_with_dict(d: dict, q: Qobj, newexcitations: int,
                               truncate, trunccount=0):
    """
    Helper function to do the tensor product between a dictionary (which
    represents a Qobj) and a Qobj.
    """
    out = {}
    mat = q.data.as_scipy().tocoo()

    # loop over nonzero elements of the qobj
    for row_idx, col_idx, val in zip(mat.row, mat.col, mat.data):
        # convert the indices to the state tuples
        row_state = tuple(q._dims[0].idx2dims(row_idx))
        col_state = tuple(q._dims[1].idx2dims(col_idx))

        # loop over the nonzero elements of the dictionary
        for (row_state_old, col_state_old), oldval in d.items():
            # concatenate the states
            row_state_new = row_state_old + row_state
            col_state_new = col_state_old + col_state

            if (sum(row_state_new) <= newexcitations
                    and sum(col_state_new) <= newexcitations):
                out[row_state_new, col_state_new] = oldval * val
            elif truncate:
                trunccount += 1
            else:
                missingstates = (str(state) for state
                                 in [row_state_new, col_state_new]
                                 if sum(state) > newexcitations)
                msg = (
                    "state " + ", ".join(missingstates) +
                    " is not in the new restricted state space. " +
                    "Set `truncate=True` to truncate these entries."
                )
                raise ValueError(msg) from None

    return (out, trunccount)


def _enr_qobj_from_dict(op_dict, dims, excitations, **kwargs):
    rowspace, row_idxlist = _get_space_and_indices(
        op_dict, dims, 0, excitations)
    colspace, col_idxlist = _get_space_and_indices(
        op_dict, dims, 1, excitations)

    op_coo = scipy.sparse.coo_matrix(
        (list(op_dict.values()), (row_idxlist, col_idxlist)),
        shape=(rowspace.size, colspace.size), dtype="complex128")

    return Qobj(op_coo, dims=[rowspace, colspace], **kwargs)


def _get_space_and_indices(op_dict, dims, dim_idx, excitations):
    if np.prod(dims[dim_idx]) == 1:  # 1-dimensional (e.g. ket)
        space = Space(list(dims[dim_idx]))
        indices = [0]*len(op_dict)
    else:
        space = EnrSpace(dims[dim_idx], excitations)
        indices = [space.dims2idx(coord_state[dim_idx])
                   for coord_state in op_dict.keys()]
    return space, indices
