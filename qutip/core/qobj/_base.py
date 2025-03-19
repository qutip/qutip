from __future__ import annotations

import numpy as np
import numbers
import functools
import scipy.sparse

import qutip
from ... import __version__
from ...settings import settings
from .. import data as _data
from ..dimensions import (
    enumerate_flat, collapse_dims_super, flatten, unflatten, Dimensions
)
from typing import Any, Literal
from qutip.typing import LayerType, DimensionLike
from numpy.typing import ArrayLike


__all__ = ["Qobj"]


def _latex_real(x):
    if not x:
        return "0"
    if not 0.001 <= abs(x) < 1000:
        base, exp = "{:.3e}".format(x).split('e')
        return base + r"\times10^{{ {:d} }}".format(int(exp))
    if abs(x - int(x)) < 0.001:
        return "{:d}".format(round(x))
    return "{:.3f}".format(x)


def _latex_complex(x):
    if abs(x.imag) < 0.001:
        return _latex_real(x.real)
    if abs(x.real) < 0.001:
        return _latex_real(x.imag) + "j"
    sign = "+" if x.imag > 0 else "-"
    return "(" + _latex_real(x.real) + sign + _latex_real(abs(x.imag)) + "j)"


def _latex_row(row, cols, data):
    if row is None:
        bits = (r"\ddots" if col is None else r"\vdots" for col in cols)
    else:
        bits = (r"\cdots" if col is None else _latex_complex(data[row, col])
                for col in cols)
    return " & ".join(bits)


class _QobjBuilder(type):
    qobjtype_to_class = {}

    @staticmethod
    def _initialize_data(arg, raw_dims, copy):
        flags = {}
        if isinstance(arg, _data.Data):
            data = arg.copy() if copy else arg
            dims = Dimensions(raw_dims or [[arg.shape[0]], [arg.shape[1]]])
        elif isinstance(arg, Qobj):
            data = arg.data.copy() if copy else arg.data
            dims = Dimensions(raw_dims or arg._dims)
            flags["isherm"] = arg._isherm
            flags["isunitary"] = arg._isunitary
        else:
            data = _data.create(arg, copy=copy)
            dims = Dimensions(
                raw_dims or [[data.shape[0]], [data.shape[1]]]
            )
        if dims.shape != data.shape:
            raise ValueError('Provided dimensions do not match the data: ' +
                             f"{dims.shape} vs {data.shape}")

        return data, dims, flags

    def __call__(
        cls,
        arg: ArrayLike | Any = None,
        dims: DimensionLike = None,
        copy: bool = True,
        superrep: str = None,
        isherm: bool = None,
        isunitary: bool = None,
        dtype: type | str = None,
    ):
        if cls is not Qobj:
            out = cls.__new__(cls)
            out.__init__(
                arg, dims,
                copy=copy, superrep=superrep,
                isherm=isherm, isunitary=isunitary
            )
            return out

        data, dims, flags = _QobjBuilder._initialize_data(arg, dims, copy)
        if (
            isinstance(arg, list)
            or dtype
            or settings.core["default_dtype_scope"] == "full"
        ):
            dtype = dtype or settings.core["default_dtype"]
            if not (dtype is None or isinstance(data, _data.to.parse(dtype))):
                data = _data.to(dtype, data)

        if isherm is not None:
            flags["isherm"] = isherm
        if isunitary is not None:
            flags["isunitary"] = isunitary
        if superrep is not None:
            dims = dims.replace_superrep(superrep)

        instance_class = _QobjBuilder.qobjtype_to_class[dims.type]

        new_qobj = instance_class.__new__(instance_class)
        new_qobj.__init__(data, dims, **flags)
        return new_qobj


def _require_equal_type(method):
    """
    Decorate a binary Qobj method to ensure both operands are Qobj and of the
    same type and dimensions.  Promote numeric scalar to identity matrices of
    the same type and shape.
    """
    @functools.wraps(method)
    def out(self, other):
        if isinstance(other, Qobj):
            if self._dims != other._dims:
                msg = (
                    "incompatible dimensions "
                    + repr(self.dims) + " and " + repr(other.dims)
                )
                raise ValueError(msg)
            return method(self, other)
        if other == 0:
            return method(self, other)
        if self._dims.issquare and isinstance(other, numbers.Number):
            scale = complex(other)
            other = Qobj(_data.identity(self.shape[0], scale,
                                        dtype=type(self.data)),
                         dims=self._dims,
                         isherm=(scale.imag == 0),
                         isunitary=(abs(abs(scale)-1) < settings.core['atol']),
                         copy=False)
            return method(self, other)
        return NotImplemented

    return out


class Qobj(metaclass=_QobjBuilder):
    """
    A class for representing quantum objects, such as quantum operators and
    states.

    The Qobj class is the QuTiP representation of quantum operators and state
    vectors. This class also implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator/state operations.  The Qobj constructor optionally takes a
    dimension ``list`` and/or shape ``list`` as arguments.

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
    _dims: Dimensions
    _data: Data
    __array_ufunc__ = None

    def __init__(
        self,
        arg: ArrayLike | Any = None,
        dims: DimensionLike = None,
        copy: bool = True,
        superrep: str = None,
        isherm: bool = None,
        isunitary: bool = None
    ):
        flags = {"isherm": isherm, "isunitary": isunitary}
        if not (isinstance(arg, _data.Data) and isinstance(dims, Dimensions)):
            arg, dims, flags = _QobjBuilder._initialize_data(arg, dims, copy)
            if isherm is None and flags["isherm"] is not None:
                isherm = flags["isherm"]
            if isunitary is None and flags["isunitary"] is not None:
                isunitary = flags["isunitary"]
            if superrep is not None:
                dims = dims.replace_superrep(superrep)

        self._data = arg
        self._dims = dims
        self._flags = {}
        if isherm is not None:
            self._flags["isherm"] = isherm
        else:
            self._flags["isherm"] = flags.get("isherm", None)
        if isunitary is not None:
            self._flags["isunitary"] = isunitary
        else:
            self._flags["isunitary"] = flags.get("isunitary", None)

    @property
    def type(self) -> str:
        return self._dims.type

    @property
    def data(self) -> _data.Data:
        return self._data

    @data.setter
    def data(self, data: _data.Data):
        if not isinstance(data, _data.Data):
            raise TypeError('Qobj data must be a data-layer format.')
        if self._dims.shape != data.shape:
            raise ValueError('Provided data do not match the dimensions: ' +
                             f"{self._dims.shape} vs {data.shape}")
        self._data = data

    @property
    def dtype(self):
        return type(self._data)

    @property
    def dims(self) -> list[list[int]] | list[list[list[int]]]:
        return self._dims.as_list()

    @dims.setter
    def dims(self, dims: list[list[int]] | list[list[list[int]]] | Dimensions):
        dims = Dimensions(dims, rep=self.superrep)
        if dims.shape != self._data.shape:
            raise ValueError('Provided dimensions do not match the data: ' +
                             f"{dims.shape} vs {self._data.shape}")
        self._dims = dims

    def copy(self) -> Qobj:
        """Create identical copy"""
        return self.__class__(
            arg=self._data,
            dims=self._dims,
            isherm=self._isherm,
            isunitary=self._isunitary,
            dtype=self.dtype,
            copy=True
        )

    def to(self, data_type: LayerType, copy: bool = False) -> Qobj:
        """
        Convert the underlying data store of this `Qobj` into a different
        storage representation.

        The different storage representations available are the "data-layer
        types" which are known to :obj:`qutip.core.data.to`.  By default, these
        are :class:`~qutip.core.data.CSR`, :class:`~qutip.core.data.Dense` and
        :class:`~qutip.core.data.Dia`, which respectively construct a
        compressed sparse row matrix, diagonal matrix and a dense one.  Certain
        algorithms and operations may be faster or more accurate when using a
        more appropriate data store.

        Parameters
        ----------
        data_type : type, str
            The data-layer type or its string alias that the data of this
            :class:`Qobj` should be converted to.

        copy : Bool
            If the data store is already in the format requested, whether the
            function should return returns `self` or a copy.

        Returns
        -------
        Qobj
            A :class:`Qobj` with the data stored in the requested format.
        """
        data_type = _data.to.parse(data_type)
        if type(self._data) is data_type and copy:
            return self.copy()
        elif type(self._data) is data_type:
            return self
        return self.__class__(
            _data.to(data_type, self._data),
            dims=self._dims,
            isherm=self._isherm,
            isunitary=self._isunitary,
            dtype=data_type,
            copy=False
        )

    @_require_equal_type
    def __add__(self, other: Qobj | complex) -> Qobj:
        if other == 0:
            return self.copy()
        return Qobj(_data.add(self._data, other._data),
                    dims=self._dims,
                    isherm=(self._isherm and other._isherm) or None,
                    copy=False)

    def __radd__(self, other: Qobj | complex) -> Qobj:
        return self.__add__(other)

    @_require_equal_type
    def __sub__(self, other: Qobj | complex) -> Qobj:
        if other == 0:
            return self.copy()
        return Qobj(_data.sub(self._data, other._data),
                    dims=self._dims,
                    isherm=(self._isherm and other._isherm) or None,
                    copy=False)

    def __rsub__(self, other: Qobj | complex) -> Qobj:
        return self.__neg__().__add__(other)

    def __neg__(self) -> Qobj:
        return Qobj(_data.neg(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def __mul__(self, other: complex) -> Qobj:
        """
        If other is a Qobj, we dispatch to __matmul__. If not, we
        check that other is a valid complex scalar, i.e., we can do
        complex(other). Otherwise, we return NotImplemented.
        """

        if isinstance(other, Qobj):
            return self.__matmul__(other)

        # We send other to mul instead of complex(other) to be more flexible.
        # The dispatcher can then decide how to handle other and return
        # TypeError if it does not know what to do with the type of other.
        try:
            out = _data.mul(self._data, other)
        except TypeError:
            return NotImplemented

        # Infer isherm and isunitary if possible
        try:
            multiplier = complex(other)
            isherm = (self._isherm and multiplier.imag == 0) or None
            isunitary = (abs(abs(multiplier) - 1) < settings.core['atol']
                         if self._isunitary else None)
        except TypeError:
            isherm = None
            isunitary = None

        return Qobj(out,
                    dims=self._dims,
                    isherm=isherm,
                    isunitary=isunitary,
                    copy=False)

    def __rmul__(self, other: complex) -> Qobj:
        # Shouldn't be here unless `other.__mul__` has already been tried, so
        # we _shouldn't_ check that `other` is `Qobj`.
        return self.__mul__(other)

    def __truediv__(self, other: complex) -> Qobj:
        return self.__mul__(1 / other)

    def __matmul__(self, other: Qobj) -> Qobj:
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other)
            except TypeError:
                return NotImplemented
        new_dims = self._dims @ other._dims

        if new_dims.type == 'scalar':
            return _data.inner(self._data, other._data)

        if self.isket and other.isbra:
            return Qobj(
                _data.matmul_outer(self._data, other._data),
                dims=new_dims,
                isunitary=False,
                copy=False
            )

        return Qobj(
            _data.matmul(self._data, other._data),
            dims=new_dims,
            isunitary=self._isunitary and other._isunitary,
            copy=False
        )

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Qobj) or self._dims != other._dims:
            return False
        # isequal uses both atol and rtol from settings.core
        return _data.isequal(self._data, other._data)

    def __and__(self, other: Qobj) -> Qobj:
        """
        Syntax shortcut for tensor:
        A & B ==> tensor(A, B)
        """
        return qutip.tensor(self, other)

    def dag(self) -> Qobj:
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        return Qobj(_data.adjoint(self._data),
                    dims=Dimensions(self._dims[0], self._dims[1]),
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def conj(self) -> Qobj:
        """Get the element-wise conjugation of the quantum object."""
        return Qobj(_data.conj(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def trans(self) -> Qobj:
        """Get the matrix transpose of the quantum operator.

        Returns
        -------
        oper : :class:`.Qobj`
            Transpose of input operator.
        """
        return Qobj(_data.transpose(self._data),
                    dims=Dimensions(self._dims[0], self._dims[1]),
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def _str_header(self):
        out = ", ".join([
            f"{self.__class__.__name__}: dims={self.dims}",
            f"shape={self._data.shape}",
            f"type={repr(self.type)}",
            f"dtype={self.dtype.__name__}",
        ])
        # TODO: Should this be here?
        if self.type in ('oper', 'super'):
            out += ", isherm=" + str(self.isherm)
        if self.issuper and self.superrep != 'super':
            out += ", superrep=" + repr(self.superrep)
        return out

    def __str__(self):
        if self.data.shape[0] * self.data.shape[0] > 100_000_000:
            # If the system is huge, don't attempt to convert to a dense matrix
            # and then to string, because it is pointless and is likely going
            # to produce memory errors. Instead print the sparse data string
            # representation.
            data = _data.to(_data.CSR, self.data).as_scipy()
        elif _data.iszero(_data.sub(self.data.conj(), self.data)):
            # TODO: that check could be slow...
            data = np.real(self.full())
        else:
            data = self.full()
        return "\n".join([self._str_header(), "Qobj data =", str(data)])

    def __repr__(self):
        # give complete information on Qobj without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    def _repr_latex_(self):
        """
        Generate a LaTeX representation of the Qobj instance. Can be used for
        formatted output in ipython notebook.
        """
        half_length = 5
        n_rows, n_cols = self.data.shape
        # Choose which rows and columns we're going to output, or None if that
        # element should be truncated.
        rows = list(range(min((half_length, n_rows))))
        if n_rows <= half_length * 2:
            rows += list(range(half_length, min((2*half_length, n_rows))))
        else:
            rows.append(None)
            rows += list(range(n_rows - half_length, n_rows))
        cols = list(range(min((half_length, n_cols))))
        if n_cols <= half_length * 2:
            cols += list(range(half_length, min((2*half_length, n_cols))))
        else:
            cols.append(None)
            cols += list(range(n_cols - half_length, n_cols))
        # Make the data array.
        data = r'$$\left(\begin{array}{cc}'
        data += r"\\".join(_latex_row(row, cols, self.data.to_array())
                           for row in rows)
        data += r'\end{array}\right)$$'
        return self._str_header() + data

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        # TODO: what happen with miss matched child class?
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)

    def __getitem__(self, ind):
        # TODO: should we require that data-layer types implement this?  This
        # isn't the right way of handling it, for sure.
        if isinstance(self._data, _data.CSR):
            data = self._data.as_scipy()
        elif isinstance(self._data, _data.Dense):
            data = self._data.as_ndarray()
        else:
            data = self._data
        try:
            out = data[ind]
            return out.toarray() if scipy.sparse.issparse(out) else out
        except TypeError:
            pass
        return data.to_array()[ind]

    def full(
        self,
        order: Literal['C', 'F'] = 'C',
        squeeze: bool = False
    ) -> np.ndarray:
        """Dense array from quantum object.

        Parameters
        ----------
        order : str {'C', 'F'}
            Return array in C (default) or Fortran ordering.
        squeeze : bool {False, True}
            Squeeze output array.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.
        """
        out = np.asarray(self.data.to_array(), order=order)
        return out.squeeze() if squeeze else out

    def data_as(self, format: str = None, copy: bool = True) -> Any:
        """Matrix from quantum object.

        Parameters
        ----------
        format : str, default: None
            Type of the output, "ndarray" for ``Dense``, "csr_matrix" for
            ``CSR``. A ValueError will be raised if the format is not
            supported.

        copy : bool {False, True}
            Whether to return a copy

        Returns
        -------
        data : numpy.ndarray, scipy.sparse.matrix_csr, etc.
            Matrix in the type of the underlying libraries.
        """
        return _data.extract(self._data, format, copy)

    def contract(self, inplace: bool = False) -> Qobj:
        """
        Contract subspaces of the tensor structure which are 1D.  Not defined
        on superoperators.  If all dimensions are scalar, a Qobj of dimension
        [[1], [1]] is returned, i.e. _multiple_ scalar dimensions are
        contracted, but one is left.

        Parameters
        ----------
        inplace: bool, optional
            If ``True``, modify the dimensions in place.  If ``False``, return
            a copied object.

        Returns
        -------
        out: :class:`.Qobj`
            Quantum object with dimensions contracted.  Will be ``self`` if
            ``inplace`` is ``True``.
        """
        if self.isket:
            sub = [x for x in self.dims[0] if x > 1] or [1]
            dims = [sub, [1]*len(sub)]
        elif self.isbra:
            sub = [x for x in self.dims[1] if x > 1] or [1]
            dims = [[1]*len(sub), sub]
        elif self.isoper or self.isoperket or self.isoperbra:
            if self.isoper:
                oper_dims = self.dims
            elif self.isoperket:
                oper_dims = self.dims[0]
            else:
                oper_dims = self.dims[1]
            if len(oper_dims[0]) != len(oper_dims[1]):
                raise ValueError("cannot parse Qobj dimensions: "
                                 + repr(self.dims))
            dims_ = [
                (x, y) for x, y in zip(oper_dims[0], oper_dims[1])
                if x > 1 or y > 1
            ] or [(1, 1)]
            dims = [[x for x, _ in dims_], [y for _, y in dims_]]
            if self.isoperket:
                dims = [dims, [1]]
            elif self.isoperbra:
                dims = [[1], dims]
        else:
            raise TypeError("not defined for superoperators")
        if inplace:
            self.dims = dims
            return self
        return Qobj(self.data.copy(), dims=dims, copy=False)

    def permute(self, order: list) -> Qobj:
        """
        Permute the tensor structure of a quantum object.  For example,

            ``qutip.tensor(x, y).permute([1, 0])``

        will give the same result as

            ``qutip.tensor(y, x)``

        and

            ``qutip.tensor(a, b, c).permute([1, 2, 0])``

        will be the same as

            ``qutip.tensor(b, c, a)``

        For regular objects (bras, kets and operators) we expect ``order`` to
        be a flat list of integers, which specifies the new order of the tensor
        product.

        For superoperators, we expect ``order`` to be something like

            ``[[0, 2], [1, 3]]``

        which tells us to permute according to [0, 2, 1, 3], and then group
        indices according to the length of each sublist.  As another example,
        permuting a superoperator with dimensions of

            ``[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]``

        by an ``order``

            ``[[0, 3], [1, 4], [2, 5]]``

        should give a new object with dimensions

            ``[[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]]``.

        Parameters
        ----------
        order : list
            List of indices specifying the new tensor order.

        Returns
        -------
        P : :class:`.Qobj`
            Permuted quantum object.
        """
        if self.type in ('bra', 'ket', 'oper'):
            structure = self.dims[1] if self.isbra else self.dims[0]
            new_structure = [structure[x] for x in order]
            if self.isbra:
                dims = [self.dims[0], new_structure]
            elif self.isket:
                dims = [new_structure, self.dims[1]]
            else:
                if self._dims[0] != self._dims[1]:
                    raise TypeError("undefined for non-square operators")
                dims = [new_structure, new_structure]
            data = _data.permute.dimensions(self.data, structure, order)
            return Qobj(data,
                        dims=dims,
                        isherm=self._isherm,
                        isunitary=self._isunitary,
                        copy=False)
        # If we've got here, we're some form of superoperator, so we work with
        # the flattened structure.
        flat_order = flatten(order)
        flat_structure = flatten(self.dims[1] if self.isoperbra
                                 else self.dims[0])
        new_structure = unflatten([flat_structure[x] for x in flat_order],
                                  enumerate_flat(order))
        if self.isoperbra:
            dims = [self.dims[0], new_structure]
        elif self.isoperket:
            dims = [new_structure, self.dims[1]]
        else:
            if self._dims[0] != self._dims[1]:
                raise TypeError("undefined for non-square operators")
            dims = [new_structure, new_structure]
        data = _data.permute.dimensions(self.data, flat_structure, flat_order)
        return Qobj(data,
                    dims=dims,
                    superrep=self.superrep,
                    copy=False)

    def tidyup(self, atol: float = None) -> Qobj:
        """
        Removes small elements from the quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup. Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper : :class:`.Qobj`
            Quantum object with small elements removed.
        """
        atol = atol or settings.core['auto_tidyup_atol']
        self.data = _data.tidyup(self.data, atol)
        return self

    # TODO: What about superoper?
    # TODO: split into each cases?
    def transform(
        self,
        inpt: list[Qobj] | ArrayLike,
        inverse: bool = False
    ) -> Qobj:
        """Basis transform defined by input array.

        Input array can be a ``matrix`` defining the transformation,
        or a ``list`` of kets that defines the new basis.

        Parameters
        ----------
        inpt : array_like
            A ``matrix`` or ``list`` of kets defining the transformation.
        inverse : bool
            Whether to return inverse transformation.

        Returns
        -------
        oper : :class:`.Qobj`
            Operator in new basis.

        Notes
        -----
        This function is still in development.
        """
        if isinstance(inpt, list) or (isinstance(inpt, np.ndarray) and
                                      inpt.ndim == 1):
            if len(inpt) != max(self.shape):
                raise TypeError(
                    'Invalid size of ket list for basis transformation')
            base = np.hstack([psi.full() for psi in inpt])
            S = _data.adjoint(_data.create(base))
        elif isinstance(inpt, Qobj) and inpt.isoper:
            S = inpt.data
        elif isinstance(inpt, np.ndarray):
            S = _data.create(inpt).conj()
        else:
            raise TypeError('Invalid operand for basis transformation')

        # transform data
        if inverse:
            if self.isket:
                data = _data.matmul(S.adjoint(), self.data)
            elif self.isbra:
                data = _data.matmul(self.data, S)
            elif self.isoper:
                data = _data.matmul(_data.matmul(S.adjoint(), self.data), S)
            else:
                raise NotImplementedError
        else:
            if self.isket:
                data = _data.matmul(S, self.data)
            elif self.isbra:
                data = _data.matmul(self.data, S.adjoint())
            elif self.isoper:
                data = _data.matmul(_data.matmul(S, self.data), S.adjoint())
            else:
                raise NotImplementedError
        return Qobj(data,
                    dims=self.dims,
                    isherm=self._isherm,
                    superrep=self.superrep,
                    copy=False)

    # TODO: split?
    def overlap(self, other: Qobj) -> complex:
        """
        Overlap between two state vectors or two operators.

        Gives the overlap (inner product) between the current bra or ket Qobj
        and and another bra or ket Qobj. It gives the Hilbert-Schmidt overlap
        when one of the Qobj is an operator/density matrix.

        Parameters
        ----------
        other : :class:`.Qobj`
            Quantum object for a state vector of type 'ket', 'bra' or density
            matrix.

        Returns
        -------
        overlap : complex
            Complex valued overlap.

        Raises
        ------
        TypeError
            Can only calculate overlap between a bra, ket and density matrix
            quantum objects.
        """
        if not isinstance(other, Qobj):
            raise TypeError("".join([
                "cannot calculate overlap with non-quantum object ",
                repr(other),
            ]))
        if (
            self.type not in ('ket', 'bra', 'oper')
            or other.type not in ('ket', 'bra', 'oper')
        ):
            msg = "only bras, kets and density matrices have defined overlaps"
            raise TypeError(msg)
        left, right = self._data.adjoint(), other.data
        if self.isoper or other.isoper:
            if not self.isoper:
                left = _data.project(left)
            if not other.isoper:
                right = _data.project(right)
            return _data.trace(_data.matmul(left, right))
        if other.isbra:
            right = right.adjoint()
        out = _data.inner(left, right, self.isket)
        if self.isket and other.isbra:
            # In this particular case, we've basically doing
            #   conj(other.overlap(self))
            # so we take care to conjugate the output.
            out = np.conj(out)
            # TODO: Is this still True?
            # This case is the only one giving different results:
            # A.overlap(B) == C
            # A.dag().overlap(B) == C
            # A.overlap(B.dag()) == conj(C)
            # A.dag().overlap(B.dag()) == C
            # B.overlap(A) == C
            # B.dag().overlap(A) == C
            # B.overlap(A.dag()) == conj(C)
            # B.dag().overlap(A.dag()) == C
        return out

    @property
    def ishp(self) -> bool:
        return False

    @property
    def iscp(self) -> bool:
        return False

    @property
    def istp(self) -> bool:
        return False

    @property
    def iscptp(self) -> bool:
        return False

    @property
    def isherm(self) -> bool:
        return False

    @property
    def _isherm(self) -> bool:
        # Weak version of `isherm`, does not compute if missing
        return self._flags.get("isherm", None)

    @property
    def isunitary(self) -> bool:
        return False

    @property
    def _isunitary(self) -> bool:
        # Weak version of `isunitary`, does not compute if missing
        return self._flags.get("isunitary", None)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the Qobj data."""
        return self._data.shape

    @property
    def isoper(self) -> bool:
        """Indicates if the Qobj represents an operator."""
        return False

    @property
    def isbra(self) -> bool:
        """Indicates if the Qobj represents a bra state."""
        return False

    @property
    def isket(self) -> bool:
        """Indicates if the Qobj represents a ket state."""
        return False

    @property
    def issuper(self) -> bool:
        """Indicates if the Qobj represents a superoperator."""
        return False

    @property
    def isoperket(self) -> bool:
        """Indicates if the Qobj represents a operator-ket state."""
        return False

    @property
    def isoperbra(self) -> bool:
        """Indicates if the Qobj represents a operator-bra state."""
        return False

    @property
    def superrep(self) -> str:
        return self._dims.superrep

    @superrep.setter
    def superrep(self, super_rep: str):
        self._dims = self._dims.replace_superrep(super_rep)

    def ptrace(self, sel: int | list[int], dtype: LayerType = None) -> Qobj:
        """
        Take the partial trace of the quantum object leaving the selected
        subspaces.  In other words, trace out all subspaces which are _not_
        passed.

        This is typically a function which acts on operators; bras and kets
        will be promoted to density matrices before the operation takes place
        since the partial trace is inherently undefined on pure states.

        For operators which are currently being represented as states in the
        superoperator formalism (i.e. the object has type `operator-ket` or
        `operator-bra`), the partial trace is applied as if the operator were
        in the conventional form.  This means that for any operator `x`,
        ``operator_to_vector(x).ptrace(0) == operator_to_vector(x.ptrace(0))``
        and similar for `operator-bra`.

        The story is different for full superoperators.  In the formalism that
        QuTiP uses, if an operator has dimensions (`dims`) of
        `[[2, 3], [2, 3]]` then it can be represented as a state on a Hilbert
        space of dimensions `[2, 3, 2, 3]`, and a superoperator would be an
        operator which acts on this joint space.  This function performs the
        partial trace on superoperators by letting the selected components
        refer to elements of the _joint_ _space_, and then returns a regular
        operator (of type `oper`).

        Parameters
        ----------
        sel : int or iterable of int
            An ``int`` or ``list`` of components to keep after partial trace.
            The selected subspaces will _not_ be reordered, no matter order
            they are supplied to `ptrace`.

        dtype : type, str
            The matrix format of output.

        Returns
        -------
        oper : :class:`.Qobj`
            Quantum object representing partial trace with selected components
            remaining.
        """
        return qutip.ptrace(self, sel, dtype)
