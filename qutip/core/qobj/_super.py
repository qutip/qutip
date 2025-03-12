from ._base import Qobj, _QobjBuilder, _require_equal_type
from ._operator import Operator
import qutip
from ..dimensions import enumerate_flat, collapse_dims_super, Dimensions
import numpy as np
from ...settings import settings
from .. import data as _data
from typing import Sequence
from qutip.typing import LayerType, DimensionLike


__all__ = []


class SuperOperator(Qobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type not in ["super"]:
            raise ValueError(
                f"Expected super operator dimensions, but got {self._dims.type}"
            )

    @property
    def issuper(self) -> bool:
        return True

    @property
    def isherm(self) -> bool:
        if self._flags.get("isherm", None) is None:
            self._flags["isherm"] = _data.isherm(self._data)
        return self._flags["isherm"]

    @isherm.setter
    def isherm(self, isherm: bool):
        self._flags["isherm"] = isherm

    @property
    def isunitary(self) -> bool:
        if self._flags.get("isunitary", None) is None:
            if not self.isoper or self._data.shape[0] != self._data.shape[1]:
                self._flags["isunitary"] = False
            else:
                cmp = _data.matmul(self._data, self._data.adjoint())
                iden = _data.identity_like(cmp)
                self._flags["isunitary"] = _data.iszero(
                    _data.sub(cmp, iden), tol=settings.core['atol']
                )
        return self._flags["isunitary"]

    @isunitary.setter
    def isunitary(self, isunitary: bool):
        self._flags["isunitary"] = isunitary

    @property
    def ishp(self) -> bool:
        if self._flags.get("ishp", None) is None:
            try:
                J = qutip.to_choi(self)
                self._flags["ishp"] = J.isherm
            except:
                self._flags["ishp"] = False

        return self._flags["ishp"]

    @property
    def iscp(self) -> bool:
        if self._flags.get("iscp", None) is None:
            # We can test with either Choi or chi, since the basis
            # transformation between them is unitary and hence preserves
            # the CP and TP conditions.
            J = self if self.superrep in ('choi', 'chi') else qutip.to_choi(self)
            # If J isn't hermitian, then that could indicate either that J is not
            # normal, or is normal, but has complex eigenvalues.  In either case,
            # it makes no sense to then demand that the eigenvalues be
            # non-negative.
            self._flags["iscp"] = (
                J.isherm
                and np.all(J.eigenenergies() >= -settings.core['atol'])
            )
        return self._flags["iscp"]

    @property
    def istp(self) -> bool:
        if self._flags.get("istp", None) is None:
            # Normalize to a super of type choi or chi.
            # We can test with either Choi or chi, since the basis
            # transformation between them is unitary and hence
            # preserves the CP and TP conditions.
            if self.issuper and self.superrep in ('choi', 'chi'):
                qobj = self
            else:
                qobj = qutip.to_choi(self)
            # Possibly collapse dims.
            if any([len(index) > 1
                    for super_index in qobj.dims
                    for index in super_index]):
                qobj = Qobj(qobj.data,
                            dims=collapse_dims_super(qobj.dims),
                            superrep=qobj.superrep,
                            copy=False)
            # We use the condition from John Watrous' lecture notes,
            # Tr_1(J(Phi)) = identity_2.
            # See: https://cs.uwaterloo.ca/~watrous/LectureNotes.html,
            # Theory of Quantum Information (Fall 2011), theorem 5.4.
            tr_oper = qobj.ptrace([0])
            self._flags["istp"] = np.allclose(
                tr_oper.full(),
                np.eye(tr_oper.shape[0]),
                atol=settings.core['atol']
            )
        return self._flags["istp"]

    @property
    def iscptp(self) -> bool:
        if (
            self._flags.get("iscp", None) is None
            and self._flags.get("istp", None) is None
        ):
            reps = ('choi', 'chi')
            q_oper = qutip.to_choi(self) if self.superrep not in reps else self
            self._flags["iscp"] = q_oper.iscp
            self._flags["istp"] = q_oper.istp
        return self.iscp and self.istp

    def dnorm(self, B: Qobj = None) -> float:
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return qutip.dnorm(self, B)

    def __call__(self, other: Qobj) -> Qobj:
        """
        Acts this Qobj on another Qobj either by left-multiplication,
        or by vectorization and devectorization, as
        appropriate.
        """
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")
        if other.type not in ["ket", "oper"]:
            raise TypeError(self.type + " cannot act on " + other.type)
        if other.isket:
            other = other.proj()
        return qutip.vector_to_operator(self @ qutip.operator_to_vector(other))

    # Method from operator are merged on case per case basis

    def _warn(f, name=None):
        import warnings
        def func(*a, **kw):
            warnings.warn(f"SuperOperator.{name} used")
            return f(*a, **kw)
        return func

    # Matrix operations, should we support them?
    # Can't be easily applied on kraus map
    # __pow__ = Operator.__pow__
    # expm = Operator.expm
    # logm = Operator.logm
    # cosm = Operator.cosm
    # cosm = Operator.cosm
    sqrtm = _warn(Operator.sqrtm, "sqrtm")  # Fidelity used for choi
    # inv = Operator.inv

    # These depend on the matrix representation.
    tr = Operator.tr
    # diag = Operator.diag

    eigenstates = Operator.eigenstates  # Could be modified to return dm
    eigenenergies = Operator.eigenenergies
    # groundstate = Operator.groundstate  # Useful for super operator?

    def dual_chan(self) -> Qobj:
        """Dual channel of quantum object representing a completely positive
        map.
        """
        # Uses the technique of Johnston and Kribs (arXiv:1102.0948), which
        # is only valid for completely positive maps.
        if not self.iscp:
            raise ValueError("Dual channels are only implemented for CP maps.")
        J = qutip.to_choi(self)
        tensor_idxs = enumerate_flat(J.dims)
        J_dual = qutip.tensor_swap(
            J,
            *(
                list(zip(tensor_idxs[0][1], tensor_idxs[0][0])) +
                list(zip(tensor_idxs[1][1], tensor_idxs[1][0]))
            )
        ).trans()
        J_dual.superrep = 'choi'
        return J_dual

    def to_choi(self):
        return qutip.to_choi(self)

    def to_super(self):
        return qutip.to_super(self)

    def to_kraus(self):
        if self.ishp:
            return qutip.to_kraus(self)
        else:
            from qutip.core.superop_reps import _generalized_kraus
            pp = [(u, v.dag()) for u, v in _generalized_kraus(self.to_choi())]
            return KrausMap.generalizedKraus(general_terms=pp)


_QobjBuilder.qobjtype_to_class["super"] = SuperOperator
