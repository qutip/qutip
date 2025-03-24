from ._base import Qobj, _QobjBuilder, _require_equal_type
from ._operator import OperatorQobj, _SquareOperator
import qutip
from ..dimensions import enumerate_flat, collapse_dims_super, Dimensions
import numpy as np
from ...settings import settings
from .. import data as _data
from typing import Sequence, Literal, Any
from qutip.typing import LayerType, DimensionLike


__all__ = []


class RecSuperOperatorQobj(Qobj):
    """
    A class for representing quantum objects that represent super operators.

    This class implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator operations.

    Parameters
    ----------
    arg: array_like, data object or :obj:`.Qobj`
        Data for vector/matrix representation of the quantum object.
    dims: list
        Dimensions of object used for tensor products.
    copy: bool
        Flag specifying whether Qobj should get a copy of the
        input data, or use the original.

    Attributes
    ----------
    data : object
        The data object storing the vector / matrix representation of the
        `Qobj`.
    dtype : type
        The data-layer type used for storing the data. The possible types are
        described in `Qobj.to <./classes.html#qutip.core.qobj.Qobj.to>`__.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form), 'choi' (Choi matrix with tr = dimension),
        or 'chi' (chi-matrix representation).
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
    ishp : bool
        Indicates if the quantum object represents a map, and if that map is
        hermicity preserving (HP).
    istp : bool
        Indicates if the quantum object represents a map, and if that map is
        trace preserving (TP).
    iscptp : bool
        Indicates if the quantum object represents a map that is completely
        positive and trace preserving (CPTP).
    isket : bool
        Indicates if the quantum object represents a ket.
    isbra : bool
        Indicates if the quantum object represents a bra.
    isoper : bool
        Indicates if the quantum object represents an operator.
    issuper : bool
        Indicates if the quantum object represents a superoperator.
    isoperket : bool
        Indicates if the quantum object represents an operator in column vector
        form.
    isoperbra : bool
        Indicates if the quantum object represents an operator in row vector
        form.
    """
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type not in ["rec_super"]:
            raise ValueError(
                "Expected super operator dimensions, "
                f"but got {self._dims.type}"
            )

    @property
    def issuper(self) -> bool:
        return True

    @property
    def ishp(self) -> bool:
        if self._flags.get("ishp", None) is None:
            try:
                J = qutip.to_choi(self)
                self._flags["ishp"] = J.isherm
            except:  # TODO: except what?
                self._flags["ishp"] = False

        return self._flags["ishp"]

    @property
    def iscp(self) -> bool:
        if self._flags.get("iscp", None) is None:
            # We can test with either Choi or chi, since the basis
            # transformation between them is unitary and hence preserves
            # the CP and TP conditions.
            if self.superrep in ('choi', 'chi'):
                J = self
            else:
                J = qutip.to_choi(self)
            # If J isn't hermitian, then that could indicate either that J is
            # not normal, or is normal, but has complex eigenvalues.  In either
            # case, it makes no sense to then demand that the eigenvalues be
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

    # These depend on the matrix representation.
    diag = OperatorQobj.diag

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
            return _generalized_kraus(self.to_choi())

    def norm(
        self,
        norm: Literal["max", "fro", "tr", "one"] = "tr",
        kwargs: dict[str, Any] = None
    ) -> float:
        """
        Norm of a quantum object.

        Default norm is the trace-norm. Other operator norms may be
        specified using the `norm` parameter.

        Parameters
        ----------
        norm : str, default: "tr"
            Which type of norm to use. Allowed values are 'tr' for the trace
            norm, 'fro' for the Frobenius norm, 'one' and 'max'.

        kwargs : dict, optional
            Additional keyword arguments to pass on to the relevant norm
            solver.  See details for each norm function in :mod:`.data.norm`.

        Returns
        -------
        norm : float
            The requested norm of the operator or state quantum object.
        """
        norm = norm or "tr"
        if norm not in {'tr', 'fro', 'one', 'max'}:
            raise ValueError(
                "matrix norm must be in {'tr', 'fro', 'one', 'max'}"
            )

        kwargs = kwargs or {}
        return {
            'tr': _data.norm.trace,
            'one': _data.norm.one,
            'max': _data.norm.max,
            'fro': _data.norm.frobenius,
        }[norm](self._data, **kwargs)


class SuperOperatorQobj(RecSuperOperatorQobj, _SquareOperator):
    """
    A class for representing quantum objects that represent super operators.

    This class implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator operations.

    Parameters
    ----------
    arg: array_like, data object or :obj:`.Qobj`
        Data for vector/matrix representation of the quantum object.
    dims: list
        Dimensions of object used for tensor products.
    copy: bool
        Flag specifying whether Qobj should get a copy of the
        input data, or use the original.

    Attributes
    ----------
    data : object
        The data object storing the vector / matrix representation of the
        `Qobj`.
    dtype : type
        The data-layer type used for storing the data. The possible types are
        described in `Qobj.to <./classes.html#qutip.core.qobj.Qobj.to>`__.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form), 'choi' (Choi matrix with tr = dimension),
        or 'chi' (chi-matrix representation).
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
    ishp : bool
        Indicates if the quantum object represents a map, and if that map is
        hermicity preserving (HP).
    istp : bool
        Indicates if the quantum object represents a map, and if that map is
        trace preserving (TP).
    iscptp : bool
        Indicates if the quantum object represents a map that is completely
        positive and trace preserving (CPTP).
    isket : bool
        Indicates if the quantum object represents a ket.
    isbra : bool
        Indicates if the quantum object represents a bra.
    isoper : bool
        Indicates if the quantum object represents an operator.
    issuper : bool
        Indicates if the quantum object represents a superoperator.
    isoperket : bool
        Indicates if the quantum object represents an operator in column vector
        form.
    isoperbra : bool
        Indicates if the quantum object represents an operator in row vector
        form.
    """
    def __init__(self, data, dims, **flags):
        Qobj.__init__(self, data, dims, **flags)
        if self._dims.type not in ["super"]:
            raise ValueError(
                "Expected super operator dimensions, "
                f"but got {self._dims.type}"
            )


_QobjBuilder.qobjtype_to_class["super"] = SuperOperatorQobj
_QobjBuilder.qobjtype_to_class["rec_super"] = RecSuperOperatorQobj
