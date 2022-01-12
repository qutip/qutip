"""The Quantum instrument (QInstrument) class, for representing the most
general form of discrete-time evolution in open quantum systems.
"""

__all__ = ['QInstrument', 'Seq', 'Par', 'Pauli', 'PauliString']

# # Design notes
#
# Instruments are indexed sets of functions from normalized density operators 
# to subnormalized (trace ∈ [0, 1]) density operators, with each index
# corresponding to a given classical measurement outcome.
#
# We represent instruments in QuTiP as dictionaries from classical measurement
# labels to Qobj instances for each quantum process making up an instrument.
# In particular, measurement labels are always assumed to be _tuples_ of zero
# or more different underlying labels, making it easy to compose instruments
# together. A quantum process representing no measurement thus is indexed by
# the zero-length (unit-like) tuple `()`, while a process representing a single
# measurement may be represented by tuples such as `(0,)`, `(1,)`, and so
# forth.

from enum import IntEnum
import itertools
import warnings
import types
import numbers
from typing import Dict, Iterable, List, Tuple, TypeVar, Union, Optional, Generic
from dataclasses import dataclass, replace
from qutip.core import Qobj

@dataclass
class Outcome:
    probability : float
    output_state : Qobj

    @classmethod
    def _from_qobj(cls, qobj):
        return cls(probability=qobj.tr(), output_state=qobj.unit())

# TODO: Need better names for these two classes!
class Pauli(IntEnum):
    I = 0
    X = 1
    Y = 3
    Z = 2

    def as_qobj(self):
        if self == Pauli.I:
            return ops.qeye(2)
        elif self == Pauli.X:
            return ops.sigmax()
        elif self == Pauli.Y:
            return ops.sigmay()
        elif self == Pauli.Z:
            return ops.sigmaz()

# FIXME: This is a hack, since the metaclass for IntEnum adds a __new__
#        only if we don't have one already.
__orig_pauli_new = Pauli.__new__
def __pauli_new(cls, value):
    if isinstance(value, str):
        return Pauli[value]
    return __orig_pauli_new(cls, value)
Pauli.__new__ = __pauli_new

_PHASES = ["+", "+i", "-", "-i"]

@dataclass(frozen=True)
class PauliString:
    phase: int # Factor of 𝑖
    op: Tuple[Pauli]

    def __init__(self, phase: Union[int, str, "PauliString"], op: Optional[List[Pauli]] = None):
        if isinstance(phase, str):
            value = type(self).parse(phase)
        elif isinstance(phase, PauliString):
            value = phase
        else:
            value = None

        object.__setattr__(self, "phase", value.phase if value is not None else phase)
        object.__setattr__(self, "op", tuple(value.op if value is not None else op))

    @classmethod
    def parse(cls, s):
        sign, s = (s[0] == "-", s[1:]) if s[0] in "+-" else (False, s)
        imag, s = (True, s[1:]) if s[0] == "i" else (False, s)
        phase = (2 if sign else 0) + (1 if imag else 0)
        op = list(map(Pauli, s))
        return cls(phase, op)

    def as_qobj(self):
        return (1j ** self.phase) * tensor([P.as_qobj() for P in self.op])

    def __repr__(self):
        return _PHASES[self.phase % 4] + "".join(P.name for P in self.op)

    def __neg__(self):
        return replace(self, phase=self.phase + 2)

    def __eq__(self, other):
        if isinstance(other, PauliString):
            return self.phase % 4 == other.phase % 4 and self.op == other.op
        elif isinstance(other, str):
            return self == PauliString(other)
        else:
            return NotImplemented

# TODO: Need better names for these two classes as well!
def _flatten_nested(t, items):
    return sum(
        (
            list(item) if isinstance(item, t) else [item]
            for item in items
        ),
        []
    )

def _flatten_singletons(t, items):
    empty = t()
    return [
        item[0] if isinstance(item, t) and len(item) == 1 else item
        for item in items
        if item != empty
    ]

def _normalize_seq_par_args(args, inner, outer):
    if len(args) == 0:
        return ()

    args = tuple(args)
    new_args = None
    while hash(args) != hash(new_args):
        if new_args is not None:
            args = new_args
        new_args = tuple(
            _flatten_singletons(inner, _flatten_nested(outer, args))
        )

    return new_args

class Seq(tuple):
    """
    A sequence of measurement outcomes.

    Automatically flattens nested levels of `Seq`; e.g.: `Seq(Seq(1, 2), 3)`
    is identical to `Seq(1, 2, 3)`. Nested singletons of `Par` are implied;
    e.g.: `Seq(Par(1,), 2)` is identical to `Seq(1, 2)`.
    """
    def __new__(cls, *iterable):
        return super().__new__(cls,
            _normalize_seq_par_args(iterable, Par, Seq)
        )

    def __repr__(self) -> str:
        return f"Seq{super().__repr__()}"

    def __add__(self, other):
        return Seq(*(super().__add__(other)))
    def __radd__(self, other):
        return Seq(*(super().__radd__(other)))

class Par(tuple):
    """
    A list of measurement outcomes extracted in parallel on distinct
    subsystems.

    Automatically flattens nested levels of `Par`; e.g.: `Par(Par(1, 2), 3)`
    is identical to `Par(1, 2, 3)`. Nested singletons of `Seq` are implied;
    e.g.: `Par(Seq(1,), 2)` is identical to `Par(1, 2)`.
    """
    def __new__(cls, *iterable):
        return super().__new__(cls,
            _normalize_seq_par_args(iterable, Seq, Par)
        )

    def __repr__(self) -> str:
        return f"Par{super().__repr__()}"

    def __add__(self, other):
        return Par(*(super().__add__(other)))
    def __radd__(self, other):
        return Par(*(super().__radd__(other)))


try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import numpy as np
import qutip.settings as settings
from qutip import __version__

def _is_iterable(value):
    try:
        _ = iter(value)
        return True
    except:
        return False

def _normalize_as_instrument(instrument_like):
    """
    """
    # Is the input already literally an instrument? Then copy and return its
    # data.
    if isinstance(instrument_like, QInstrument):
        return instrument_like._processes

    # Is the instrument already in the right form (that is, dictionary from
    # tuples to [Qobj | QInstrument] instances)?
    elif isinstance(instrument_like, dict):
        # Assume that the input is already a dict from measurement labels to
        # Qobj instances, but that the labels may not be tuples, and the Qobj
        # values may need to be promoted to super.
        processes = {}
        for label, value in instrument_like.items():
            label = label if isinstance(label, Seq) else Seq(label, )
            if isinstance(value, QInstrument):
                for inner_label, inner_value in value._processes.items():
                    processes[Seq(label, inner_label)] = inner_value
            else:
                if isinstance(value, Qobj) and not value.type == "super":
                    value = sr.to_super(value)
                processes[label] = value

        return processes

    # If we have a single Qobj, promote it to super if needed.
    elif isinstance(instrument_like, Qobj):
        return {
            Seq():
                instrument_like
                if instrument_like.type == "super" else
                sr.to_super(instrument_like)
        }

    # We may also have an iterable, in which case we try to promote each entry
    # and return integer indices.
    # TODO: Only go down this branch if an iterable of Qobj values!
    elif _is_iterable(instrument_like):
        values = list(instrument_like)
        return {
            Seq(idx, ):
                value
                if value.type == "super" else
                sr.to_super(value)
            for idx, value in enumerate(instrument_like)
        }

    # TODO: Try to promote anything else to a single Qobj.

    else:
        raise TypeError(f"Value {instrument_like} was not instrument-like.")

def _require_consistant_dims(processes):
    dims = None
    for process in processes:
        if dims is None:
            dims = process.dims
        else:
            if process.dims != dims:
                raise ValueError(f"Dimensions {process.dims} are not consistent with dimensions {dims}.")

def _ensure_instrument(instrument_like):
    return instrument_like if isinstance(instrument_like, QInstrument) else QInstrument(instrument_like)


class QInstrument(object):
    """TODO
    """
    __array_priority__ = 100  # sets Qobj priority above numpy arrays
    # Disable ufuncs from acting directly on Qobj. This is necessary because we
    # define __array__.
    __array_ufunc__ = None

    def __init__(self, input=None):
        """
        TODO
        """
        self._processes = _normalize_as_instrument(input)
        _require_consistant_dims(self._processes.values())

    @property
    def dims(self):
        # When constructing, we ensured that all dims are consistant, so we
        # only need to check the first process.
        return next(iter(self._processes.values())).dims

    @property
    def outcome_space(self):
        return self._processes.keys()

    @property
    def n_outcomes(self):
        return len(self._processes)

    @property
    def nonselective_process(self):
        values = list(self._processes.values())
        return sum(values[1:], values[0])

    @property
    def iscp(self):
        return all(process.iscp for process in self._processes.values()) and self.nonselective_process.iscp

    @property
    def ishp(self):
        return all(process.ishp for process in self._processes.values()) and self.nonselective_process.ishp

    @property
    def istp(self):
        # Only check the combined channel!
        return self.nonselective_process.istp
    
    @property
    def iscptp(self):
        # Don't check tp on constiuant channels!
        return all(process.iscp for process in self._processes.values()) and self.nonselective_process.iscptp

    def copy(self):
        """Create identical copy"""
        return (type(self))(input=self)

    def __len__(self):
        return self.n_outcomes

    @staticmethod
    def _compose(left, right):
        # TODO: Check if scalars work?
        try:
            left = _ensure_instrument(left)
            right = _ensure_instrument(right)
        except TypeError:
            return NotImplemented

        return QInstrument({
            # Reverse the ordering of labels, since 𝐵𝐴 means "𝐴 then 𝐵."
            Seq(right_label, left_label): prod
            for (left_label, left_process), (right_label, right_process)
            in itertools.product(left._processes.items(), right._processes.items())
            # Chop out trace-annhilating processes (those processes that
            # correspond to measurement outcomes that cannot possibly occur).
            if not (prod := (left_process * right_process)).is_trace_annhilating
        })

    def __mul__(self, other):
        """
        MULTIPLICATION with Qobj on LEFT [ ex. Qobj*4 ]
        """
        return self._compose(self, other)

    def __rmul__(self, other):
        """
        MULTIPLICATION with Qobj on RIGHT [ ex. 4*Qobj ]
        """
        return self._compose(other, self)

    def __getitem__(self, ind):
        """
        GET qobj elements.
        """
        return self._processes[ind]

    def __eq__(self, other):
        """
        EQUALITY operator.
        """
        # TODO

    def __ne__(self, other):
        """
        INEQUALITY operator.
        """
        return not (self == other)

    def __pow__(self, n):  # calculates powers of Qobj
        """
        POWER operation.
        """
        # TODO: could optimize to use binary exponentiation.
        acc = self
        for _ in range(1, n):
            acc *= self
        return acc

    # The next few operator overload methods allow `A & B` to mean
    # 𝐴 ⊗ 𝐵 and `A ^ n` to mean ⊗ᵢ₌₁ⁿ 𝐴.
    def __and__(self, other):
        return tensor(self, other)

    def __rand__(self, other):
        return tensor(other, self)

    def __xor__(self, n):
        return tensor([self] * n)

    def __str__(self):
        return repr(self) # TODO

    def __repr__(self):
        return f"QInstrument id={id(self):0x} {{\n    dims {self.dims}\n    outcomes {' '.join(map(str, self.outcome_space))}\n}}"

    def __call__(self, other):
        # TODO: Confirm that other is type=ket or type=oper.
        return {
            label: Outcome._from_qobj(process(other))
            for label, process in self._processes.items()
        }

    def sample(self, other):
        pr_table = list(self(other).items())
        idx_outcome = np.random.choice(len(pr_table), p=[outcome.probability for (label, outcome) in pr_table])
        return pr_table[idx_outcome][0], pr_table[idx_outcome][1].output_state

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)

    def _repr_latex_(self):
        # TODO
        pass

    def __array__(self, *arg, **kwarg):
        """Numpy array from Qobj
        For compatibility with np.array
        """
        # TODO

    def unit(self, inplace=False,
             norm=None, sparse=False,
             tol=0, maxiter=100000):
       # TODO
       pass

    def tidyup(self, atol=settings.core['auto_tidyup_atol']):
        """Removes small elements from the quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup. Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Quantum object with small elements removed.

        """
        # TODO

    def with_finite_visibility(self, eta):
        shape = next(iter(self._processes.values())).shape
        eye = (1 - eta) * Qobj(np.eye(*shape), dims=self.dims) / self.n_outcomes
        return type(self)({
            label: eta * process + eye
            for label, process in self._processes.items()
        })

    def complete(self):
        if self.istp:
            return type(self)(self._processes)

        if Seq("⊥") in self._processes:
            raise ValueError("Instrument already has a ⊥ outcome.")

        shape = next(iter(self._processes.values())).shape
        eye = Qobj(np.eye(*shape), dims=self.dims)
        processes = self._processes.copy()
        processes[Seq("⊥")] = eye - self.nonselective_process
        return type(self)(processes)

    def reindex(self):
        return type(self)({
            Seq(idx): process
            for (idx, (label, process)) in
            enumerate(sorted(self._processes.items(), key=lambda item: item[0]))
        })

    def if_(self, conditions: Dict):
        """
        Returns a new instrument that applies one of a given set of instruments
        conditioned on the outcome of this instrument.
        """
        # Normalize conditions to use Seq(label).
        conditions = {
            Seq(label): condition
            for label, condition in conditions.items()
        }

        processes = {}
        for label, process in self._processes.items():
            label = Seq(label)
            if label in conditions:
                processes[label] = conditions[label] * process
            else:
                # Fall back to using __eq__.
                matches = [cond_label for cond_label in conditions.keys() if cond_label == label]
                if len(matches) == 1:
                    processes[label] = conditions[matches[0]] * process
                else:
                    processes[label] = process

        return type(self)(processes)

    def ptrace(self, sel, sparse=None):
        return type(self)({
            label: process.ptrace(sel, sparse)
            for label, process in self._processes.items()
        })

## FACTORY FUNCTIONS ##

def basis_measurement(N=2):
    return QInstrument([
        qutip.states.projection(N, idx, idx)
        for idx in range(N)
    ])

# TODO: Change to list[Pauli], use _ensure_pauli to promote to string.
def pauli_measurement(pauli: Optional[Union[PauliString, str]] = None):
    """
    Returns an instrument that performs a half-space measurement on a
    given Pauli operator.
    """
    # If pauli isn't an instance of PauliString, try to promote it.
    pauli = PauliString(pauli) if isinstance(pauli, str) else pauli
    pauli = replace(pauli if pauli is not None else PauliString("+Z"), phase=0)
    op = (pauli).as_qobj()
    eye = ops.qeye(op.dims[0])
    return QInstrument({
        Seq(pauli): (op + eye) / 2,
        Seq(-pauli): (op - eye) / 2
    })

## INTERNAL UTILITY FUNCTIONS ##

def _instrument_tensor(qlist):
    terms = list(itertools.product(*[_ensure_instrument(q)._processes.items() for q in qlist]))

    return QInstrument({
        Par(*labels): super_tensor(*processes)
        for labels, processes in
        itertools.starmap(zip, terms)
    })

# TRAILING IMPORTS
# We do a few imports here to avoid circular dependencies.
import qutip.core.superop_reps as sr
import qutip.core.tensor as tensor
from .core.tensor import super_tensor
import qutip.core.operators as ops
import qutip.core.metrics as mts
import qutip.core.states
import qutip.core.superoperator
