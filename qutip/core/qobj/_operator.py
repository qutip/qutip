from ._base import Qobj, _QobjBuilder
from ...settings import settings
import numpy as np
from qutip.typing import LayerType
import qutip
from typing import Any, Literal
import numbers
from .. import data as _data
from ..dimensions import enumerate_flat, collapse_dims_super
import warnings
from typing import Any, Literal


__all__ = []


class _SquareOperator(Qobj):
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

    def __pow__(self, n: int, m=None) -> Qobj:
        # calculates powers of Qobj
        if (
            self._dims[0] != self._dims[1]
            or m is not None
            or not isinstance(n, numbers.Integral)
            or n < 0
        ):
            return NotImplemented
        return Qobj(_data.pow(self._data, n),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def dnorm(self, B: Qobj = None) -> float:
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond
            norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return qutip.dnorm(self, B)

    def expm(self, dtype: LayerType = None) -> Qobj:
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Parameters
        ----------
        dtype : type
            The data-layer type that should be output.

        Returns
        -------
        oper : :class:`.Qobj`
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.
        """
        if dtype is None and isinstance(self.data, (_data.CSR, _data.Dia)):
            dtype = _data.Dense
        return Qobj(_data.expm(self._data, dtype=dtype),
                    dims=self._dims,
                    isherm=self._isherm,
                    copy=False)

    def logm(self) -> Qobj:
        """Matrix logarithm of quantum operator.

        Input operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Logarithm of the quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.
        """
        return Qobj(_data.logm(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    copy=False)

    def sqrtm(
        self,
        sparse: bool = False,
        tol: float = 0,
        maxiter: int = 100000
    ) -> Qobj:
        """
        Sqrt of a quantum operator.  Operator must be square.

        Parameters
        ----------
        sparse : bool
            Use sparse eigenvalue/vector solver.
        tol : float
            Tolerance used by sparse solver (0 = machine precision).
        maxiter : int
            Maximum number of iterations used by sparse solver.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix square root of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        return Qobj(_data.sqrtm(self._data),
                    dims=self._dims,
                    copy=False)

    def cosm(self) -> Qobj:
        """Cosine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix cosine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.

        """
        return 0.5 * ((1j * self).expm() + (-1j * self).expm())

    def sinm(self) -> Qobj:
        """Sine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix sine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.
        """
        return -0.5j * ((1j * self).expm() - (-1j * self).expm())

    def inv(self, sparse: bool = False) -> Qobj:
        """Matrix inverse of a quantum operator

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix inverse of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.
        """
        if isinstance(self.data, _data.CSR) and not sparse:
            data = _data.to(_data.Dense, self.data)
        else:
            data = self.data

        return Qobj(_data.inv(data),
                    dims=[self._dims[1], self._dims[0]],
                    copy=False)

    def check_herm(self) -> bool:
        """Check if the quantum object is hermitian.

        Returns
        -------
        isherm : bool
            Returns the new value of isherm property.
        """
        self.isherm = None
        return self.isherm

    def tr(self) -> complex:
        """Trace of a quantum object.

        Returns
        -------
        trace : float
            Returns the trace of the quantum object.

        """
        out = _data.trace(self._data)
        # This ensures that trace can return something that is not a number
        # such as a `tensorflow.Tensor` in qutip-tensorflow.
        if settings.core["auto_real_casting"] and self.isherm:
            out = out.real
        return out

    def eigenstates(
        self,
        sparse: bool = False,
        sort: Literal["low", "high"] = 'low',
        eigvals: int = 0,
        tol: float = 0,
        maxiter: int = 100000,
        phase_fix: int = None
    ) -> tuple[np.ndarray, list[Qobj]]:
        """Eigenstates and eigenenergies.

        Eigenstates and eigenenergies are defined for operators and
        superoperators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver

        sort : str
            Sort eigenvalues (and vectors) 'low' to high, or 'high' to low.

        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.

        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.

        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        phase_fix : int, None
            If not None, set the phase of each kets so that ket[phase_fix,0]
            is real positive.

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        eigvecs : array
            Array of quantum operators representing the oprator eigenkets.
            Order of eigenkets is determined by order of eigenvalues.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        if isinstance(self.data, _data.CSR) and sparse:
            evals, evecs = _data.eigs_csr(self.data,
                                          isherm=self._isherm,
                                          sort=sort, eigvals=eigvals, tol=tol,
                                          maxiter=maxiter)
        elif isinstance(self.data, (_data.CSR, _data.Dia)):
            evals, evecs = _data.eigs(_data.to(_data.Dense, self.data),
                                      isherm=self._isherm,
                                      sort=sort, eigvals=eigvals)
        else:
            evals, evecs = _data.eigs(self.data, isherm=self._isherm,
                                      sort=sort, eigvals=eigvals)

        if self.type == 'super':
            new_dims = [self._dims[0], [1]]
        else:
            new_dims = [self._dims[0], [1]]
        ekets = np.empty((evecs.shape[1],), dtype=object)
        ekets[:] = [Qobj(vec, dims=new_dims, copy=False)
                    for vec in _data.split_columns(evecs, False)]
        norms = np.array([ket.norm() for ket in ekets])
        if phase_fix is None:
            phase = np.array([1] * len(ekets))
        else:
            phase = np.array([np.abs(ket[phase_fix, 0]) / ket[phase_fix, 0]
                              if ket[phase_fix, 0] else 1
                              for ket in ekets])
        return evals, ekets / norms * phase

    def eigenenergies(
        self,
        sparse: bool = False,
        sort: Literal["low", "high"] = 'low',
        eigvals: int = 0,
        tol: float = 0,
        maxiter: int = 100000,
    ) -> np.ndarray:
        """Eigenenergies of a quantum object.

        Eigenenergies (eigenvalues) are defined for operators or superoperators
        only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        sort : str
            Sort eigenvalues 'low' to high, or 'high' to low.
        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.
        tol : float
            Tolerance used by sparse Eigensolver (0=machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        # TODO: consider another way of handling the dispatch here.
        if isinstance(self.data, _data.CSR) and sparse:
            return _data.eigs_csr(self.data,
                                  vecs=False,
                                  isherm=self._isherm,
                                  sort=sort, eigvals=eigvals,
                                  tol=tol, maxiter=maxiter)
        elif isinstance(self.data, (_data.CSR, _data.Dia)):
            return _data.eigs(_data.to(_data.Dense, self.data),
                              vecs=False, isherm=self._isherm,
                              sort=sort, eigvals=eigvals)

        return _data.eigs(self.data,
                          vecs=False,
                          isherm=self._isherm, sort=sort, eigvals=eigvals)

    def groundstate(
        self,
        sparse: bool = False,
        tol: float = 0,
        maxiter: int = 100000,
        safe: bool = True
    ) -> tuple[float, Qobj]:
        """Ground state Eigenvalue and Eigenvector.

        Defined for quantum operators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).
        safe : bool (default=True)
            Check for degenerate ground state

        Returns
        -------
        eigval : float
            Eigenvalue for the ground state of quantum operator.
        eigvec : :class:`.Qobj`
            Eigenket for the ground state of quantum operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        eigvals = 2 if safe else 1
        evals, evecs = self.eigenstates(sparse=sparse, eigvals=eigvals,
                                        tol=tol, maxiter=maxiter)

        if safe:
            tol = tol or settings.core['atol']
            # This tol should be less strick than the tol for the eigensolver
            # so it's numerical errors are not seens as degenerate states.
            if (evals[1]-evals[0]) <= 10 * tol:
                warnings.warn("Ground state may be degenerate.", UserWarning)
        return evals[0], evecs[0]

    def trunc_neg(self, method: Literal["clip", "sgs"] = "clip") -> Qobj:
        """Truncates negative eigenvalues and renormalizes.

        Returns a new Qobj by removing the negative eigenvalues
        of this instance, then renormalizing to obtain a valid density
        operator.

        Parameters
        ----------
        method : str
            Algorithm to use to remove negative eigenvalues. "clip"
            simply discards negative eigenvalues, then renormalizes.
            "sgs" uses the SGS algorithm (doi:10/bb76) to find the
            positive operator that is nearest in the Shatten 2-norm.

        Returns
        -------
        oper : :class:`.Qobj`
            A valid density operator.
        """
        if not self.isherm:
            raise ValueError("Must be a Hermitian operator to remove negative "
                             "eigenvalues.")
        if method not in ('clip', 'sgs'):
            raise ValueError("Method {} not recognized.".format(method))

        eigvals, eigstates = self.eigenstates()
        if all(eigval >= 0 for eigval in eigvals):
            # All positive, so just renormalize.
            return self.unit()
        idx_nonzero = eigvals != 0
        eigvals = eigvals[idx_nonzero]
        eigstates = eigstates[idx_nonzero]

        if method == 'clip':
            eigvals[eigvals < 0] = 0
        elif method == 'sgs':
            eigvals = eigvals[::-1]
            eigstates = eigstates[::-1]
            acc = 0.0
            n_eigs = len(eigvals)
            for idx in reversed(range(n_eigs)):
                if eigvals[idx] + acc / (idx + 1) >= 0:
                    break
                acc += eigvals[idx]
                eigvals[idx] = 0.0
            eigvals[:idx+1] += acc / (idx + 1)
        out_data = _data.zeros(*self.shape)
        for value, state in zip(eigvals, eigstates):
            if value:
                # add in 3-argument form is fused-add-multiply
                out_data = _data.add(out_data,
                                     _data.project(state.data),
                                     value)
        out_data = _data.mul(out_data, 1/_data.norm.trace(out_data))
        return Qobj(out_data, dims=self._dims, isherm=True, copy=False)

    def purity(self) -> complex:
        """Calculate purity of a quantum object.

        Returns
        -------
        state_purity : float
            Returns the purity of a quantum object.
            For a pure state, the purity is 1.
            For a mixed state of dimension `d`, 1/d<=purity<1.

        """
        if self.type in "super":
            raise TypeError('purity is only defined for states.')
        return _data.trace(_data.matmul(self._data, self._data)).real


class RecOperatorQobj(Qobj):
    """
    A class for representing quantum objects that represent operators.

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
        if self._dims.type not in ["rec_oper"]:
            raise ValueError(
                f"Expected operator dimensions, but got {self._dims.type}"
            )

    @property
    def isoper(self) -> bool:
        return True

    @property
    def ishp(self) -> bool:
        return True

    @property
    def iscp(self) -> bool:
        return True

    @property
    def istp(self) -> bool:
        return self.isunitary

    @property
    def iscptp(self) -> bool:
        return self.isunitary

    def matrix_element(self, bra: Qobj, ket: Qobj) -> Qobj:
        """Calculates a matrix element.

        Gives the matrix element for the quantum object sandwiched between a
        `bra` and `ket` vector.

        Parameters
        ----------
        bra : :class:`.Qobj`
            Quantum object of type 'bra' or 'ket'

        ket : :class:`.Qobj`
            Quantum object of type 'ket'.

        Returns
        -------
        elem : complex
            Complex valued matrix element.

        Notes
        -----
        It is slightly more computationally efficient to use a ket
        vector for the 'bra' input.

        """
        if bra.type not in ('bra', 'ket') or ket.type not in ('bra', 'ket'):
            msg = "Can only calculate matrix elements between a bra and a ket."
            raise TypeError(msg)
        left, op, right = bra.data, self.data, ket.data
        if ket.isbra:
            right = right.adjoint()
        return _data.inner_op(left, op, right, bra.isket)

    def __call__(self, other: Qobj) -> Qobj:
        """
        Acts this Qobj on another Qobj either by left-multiplication,
        or by vectorization and devectorization, as
        appropriate.
        """
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")
        if other.type not in ["ket"]:
            raise TypeError(self.type + " cannot act on " + other.type)
        return self.__matmul__(other)

    def diag(self) -> np.ndarray:
        """Diagonal elements of quantum object.

        Returns
        -------
        diags : array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.
        """
        # TODO: add a `diagonal` method to the data layer?
        out = _data.to(_data.CSR, self.data).as_scipy().diagonal()
        if settings.core["auto_real_casting"] and self.isherm:
            out = np.real(out)
        return out

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

    def unit(
        self,
        inplace: bool = False,
        norm: Literal["l2", "max", "fro", "tr", "one"] = "tr",
        kwargs: dict[str, Any] = None
    ) -> Qobj:
        """
        Operator or state normalized to unity.  Uses norm from Qobj.norm().

        Parameters
        ----------
        inplace : bool
            Do an in-place normalization
        norm : str
            Requested norm for states / operators.
        kwargs : dict
            Additional key-word arguments to be passed on to the relevant norm
            function (see :meth:`.norm` for more details).

        Returns
        -------
        obj : :class:`.Qobj`
            Normalized quantum object.  Will be the `self` object if in place.
        """
        norm_ = self.norm(norm=norm, kwargs=kwargs)
        if inplace:
            self.data = _data.mul(self.data, 1 / norm_)
            self.isherm = self._isherm if norm_.imag == 0 else None
            self.isunitary = (
                self._isunitary
                if abs(norm_) - 1 < settings.core['atol']
                else None
            )
            out = self
        else:
            out = self / norm_
        return out


class OperatorQobj(RecOperatorQobj, _SquareOperator):
    """
    A class for representing quantum objects that represent operators.

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
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
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
        if self._dims.type not in ["oper"]:
            raise ValueError(
                f"Expected operator dimensions, but got {self._dims.type}"
            )


class ScalarQobj(OperatorQobj):
    """
    A class for representing quantum objects that represent scalar quantum
    object.

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
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
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
    # Scalar can be anything
    def __init__(self, data, dims, **flags):
        Qobj.__init__(self, data, dims, **flags)
        if self._dims.type not in ["scalar"]:
            raise ValueError(
                f"Expected scalar dimensions, but got {self._dims.type}"
            )

    @property
    def issuper(self) -> bool:
        """Indicates if the Qobj represents a superoperator."""
        return self._dims.issuper

    @property
    def isket(self) -> bool:
        return not self.issuper

    @property
    def isbra(self) -> bool:
        return not self.issuper

    @property
    def isoperket(self) -> bool:
        return self.issuper

    @property
    def isoperbra(self) -> bool:
        return self.issuper


_QobjBuilder.qobjtype_to_class["scalar"] = ScalarQobj
_QobjBuilder.qobjtype_to_class["oper"] = OperatorQobj
_QobjBuilder.qobjtype_to_class["rec_oper"] = RecOperatorQobj
