from .dimensions import Space
from .states import state_number_enumerate
from . import data as _data
from . import Qobj, qdiags
import numpy as np
import scipy.sparse
from .. import settings


__all__ = ['enr_state_dictionaries', 'enr_fock',
           'enr_thermal_dm', 'enr_destroy', 'enr_identity']


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
    dtype = dtype or settings.core["default_dtype"] or _data.Dense
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
                copy=False)


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
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
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
    dtype = dtype or settings.core["default_dtype"] or _data.CSR
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
        Qobj(a, dims=enr_dims, isunitary=False, isherm=False).to(dtype)
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
    dtype = dtype or settings.core["default_dtype"] or _data.Dia
    dims = EnrSpace(dims, excitations)
    return Qobj(_data.identity[dtype](dims.size),
                dims=[dims, dims],
                isherm=True,
                isunitary=True,
                copy=False)
