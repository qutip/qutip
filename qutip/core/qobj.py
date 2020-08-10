# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""The Quantum Object (Qobj) class, for representing quantum states and
operators, and related functions.
"""

__all__ = [
    'Qobj', 'isbra', 'isket', 'isoper', 'issuper', 'isoperbra', 'isoperket',
    'isherm', 'ptrace',
]

import functools
import numbers

import numpy as np
import scipy.sparse

from .. import __version__, settings
from . import data as _data
from .dimensions import (
    type_from_dims, enumerate_flat, collapse_dims_super, flatten, unflatten,
)


_ADJOINT_TYPE_LOOKUP = {
    'oper': 'oper',
    'super': 'super',
    'ket': 'bra',
    'bra': 'ket',
    'operator-ket': 'operator-bra',
    'operator-bra': 'operator-ket',
}

_MATMUL_TYPE_LOOKUP = {
    ('oper', 'ket'): 'ket',
    ('oper', 'oper'): 'oper',
    ('ket', 'bra'): 'oper',
    ('bra', 'oper'): 'bra',
    ('super', 'super'): 'super',
    ('super', 'operator-ket'): 'operator-ket',
    ('operator-bra', 'super'): 'operator-bra',
    ('operator-ket', 'operator-bra'): 'super',
}

_NORM_FUNCTION_LOOKUP = {
    'tr': _data.norm.trace_csr,
    'one': _data.norm.one_csr,
    'max': _data.norm.max_csr,
    'fro': _data.norm.frobenius_csr,
    'l2': _data.norm.l2_csr,
}
_NORM_ALLOWED_MATRIX = {'tr', 'fro', 'one', 'max'}
_NORM_ALLOWED_VECTOR = {'l2', 'max'}

_CALL_ALLOWED = {
    ('super', 'oper'),
    ('super', 'ket'),
    ('oper', 'ket'),
}


def isbra(x):
    return isinstance(x, Qobj) and x.type == 'bra'


def isket(x):
    return isinstance(x, Qobj) and x.type == 'ket'


def isoper(x):
    return isinstance(x, Qobj) and x.type == 'oper'


def isoperbra(x):
    return isinstance(x, Qobj) and x.type == 'operator-bra'


def isoperket(x):
    return isinstance(x, Qobj) and x.type == 'operator-ket'


def issuper(x):
    return isinstance(x, Qobj) and x.type == 'super'


def isherm(x):
    return isinstance(x, Qobj) and x.isherm


def _tidyup(method):
    """
    Wrap the input method to automatically tidy up small values in the output,
    if the relevant QuTiP global setting is there.
    """
    @functools.wraps(method)
    def out(*args, **kwargs):
        out = method(*args, **kwargs)
        if isinstance(out, Qobj) and settings.auto_tidyup:
            out.tidyup()
        return out
    return out


def _require_equal_type(method):
    """
    Decorate a binary Qobj method to ensure both operands are Qobj and of the
    same type and dimensions.  Promote numeric scalar to identity matrices of
    the same type and shape.
    """
    @functools.wraps(method)
    def out(self, other):
        if (
            self.type in ('oper', 'super')
            and self.dims[0] == self.dims[1]
            and isinstance(other, numbers.Number)
        ):
            scale = complex(other)
            other = Qobj(_data.csr.identity(self.shape[0], scale),
                         dims=self.dims,
                         type=self.type,
                         superrep=self.superrep,
                         isherm=(scale.imag == 0),
                         isunitary=(abs(abs(scale) - 1) < settings.atol),
                         copy=False)
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other, type=self.type)
            except TypeError:
                return NotImplemented
        if self.dims != other.dims:
            msg = (
                "incompatible dimensions "
                + repr(self.dims) + " and " + repr(other.dims)
            )
            raise ValueError(msg)
        if self.type != other.type:
            msg = "incompatible types " + self.type + " and " + other.type
            raise ValueError(msg)
        if self.superrep != other.superrep:
            msg = (
                "incompatible superoperator representations"
                + self.superrep + " and " + other.superrep
            )
            raise ValueError(msg)
        return method(self, other)
    return out


def _latex_real(x):
    if not x:
        return "0"
    if not 0.001 <= abs(x) < 1000:
        base, exp = "{:.3e}".format(x).split('e')
        return base + r"\times10^{{ {:d} }}".format(int(exp))
    if abs(x - int(x)) < 0.001:
        return "{:d}".format(x)
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


class Qobj:
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
    inpt: array_like
        Data for vector/matrix representation of the quantum object.
    dims: list
        Dimensions of object used for tensor products.
    type: {'bra', 'ket', 'oper', 'operator-ket', 'operator-bra', 'super'}
        The type of quantum object to be represented.
    shape: list
        Shape of underlying data structure (matrix shape).
    copy: bool
        Flag specifying whether Qobj should get a copy of the
        input data, or use the original.


    Attributes
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form) or 'choi' (Choi matrix with tr = dimension).
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

    Methods
    -------
    copy()
        Create copy of Qobj
    conj()
        Conjugate of quantum object.
    cosm()
        Cosine of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    dnorm()
        Diamond norm of quantum operator.
    dual_chan()
        Dual channel of quantum object representing a CP map.
    eigenenergies(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies (eigenvalues) of a quantum object.
    eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies and eigenstates of quantum object.
    expm()
        Matrix exponential of quantum object.
    full(order='C')
        Returns dense array of quantum object `data` attribute.
    groundstate(sparse=False, tol=0, maxiter=100000)
        Returns eigenvalue and eigenket for the groundstate of a quantum
        object.
    inv()
        Return a Qobj corresponding to the matrix inverse of the operator.
    matrix_element(bra, ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns norm of a ket or an operator.
    permute(order)
        Returns composite qobj with indices reordered.
    proj()
        Computes the projector for a ket or bra vector.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
    sinm()
        Sine of quantum object.
    sqrtm()
        Matrix square root of quantum object.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    tr()
        Trace of quantum object.
    trans()
        Transpose of quantum object.
    transform(inpt, inverse=False)
        Performs a basis transformation defined by `inpt` matrix.
    trunc_neg(method='clip')
        Removes negative eigenvalues and returns a new Qobj that is
        a valid density operator.
    unit(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns normalized quantum object.

    """
    def _initialize_data(self, arg, dims, copy):
        if isinstance(arg, _data.Data):
            self.dims = dims or [[arg.shape[0]], [arg.shape[1]]]
            self._data = arg.copy() if copy else arg
        elif isinstance(arg, Qobj):
            self.dims = dims or arg.dims.copy()
            self._data = arg.data.copy() if copy else arg.data
        elif arg is None or isinstance(arg, numbers.Number):
            self.dims = dims or [[1], [1]]
            self._data = _data.csr.identity(1, scale=complex(arg or 0))
        else:
            self._data = _data.create(arg)
            self.dims = dims or [[self._data.shape[0]], [self._data.shape[1]]]

    def __init__(self, arg=None, dims=None, type=None,
                 copy=True, superrep=None, isherm=None, isunitary=None):
        self._initialize_data(arg, dims, copy)
        self.type = type or type_from_dims(self.dims)
        self._isherm = isherm
        self._isunitary = isunitary

        if self.type == 'super' and type_from_dims(self.dims) == 'oper':
            if self._data.shape[0] != self._data.shape[1]:
                raise ValueError("".join([
                    "cannot build superoperator from nonsquare data of shape ",
                    repr(self._data.shape),
                ]))
            root = int(np.sqrt(self._data.shape[0]))
            if root * root != self._data.shape[0]:
                raise ValueError("".join([
                    "cannot build superoperator from nonsquare subspaces ",
                    "of size ",
                    repr(self._data.shape[0]),
                ]))
            self.dims = [[[root]]*2]*2
        if self.type == 'super':
            superrep = superrep or 'super'
        self.superrep = superrep

    def copy(self):
        """Create identical copy"""
        return Qobj(arg=self._data,
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=True)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, _data.Data):
            raise TypeError('Qobj data must be a data-layer format.')
        self._data = data

    @_tidyup
    @_require_equal_type
    def __add__(self, other):
        isherm = (self._isherm and other._isherm) or None
        return Qobj(_data.add_csr(self._data, other._data),
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=isherm,
                    copy=False)

    def __radd__(self, other):
        return self.__add__(other)

    @_tidyup
    @_require_equal_type
    def __sub__(self, other):
        isherm = (self._isherm and other._isherm) or None
        return Qobj(_data.sub_csr(self._data, other._data),
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=isherm,
                    copy=False)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    @_tidyup
    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return self.__matmul__(other)
        multiplier = complex(other)
        isherm = (self._isherm and multiplier.imag == 0) or None
        isunitary = (self._isunitary and abs(multiplier) == 1) or None
        return Qobj(_data.mul_csr(self._data, multiplier),
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=isherm,
                    isunitary=isunitary,
                    copy=False)

    def __rmul__(self, other):
        # Shouldn't be here unless `other.__mul__` has already been tried, so
        # we _shouldn't_ check that `other` is `Qobj`.
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self.__mul__(complex(other))

    @_tidyup
    def __matmul__(self, other):
        if not isinstance(other, Qobj):
            try:
                other = Qobj(other)
            except TypeError:
                return NotImplemented
        if self.dims[1] != other.dims[0]:
            raise TypeError("".join([
                "incompatible dimensions ",
                repr(self.dims),
                " and ",
                repr(other.dims),
            ]))
        if self.issuper and other.issuper and self.superrep != other.superrep:
            raise TypeError("".join([
                "incompatible superoperator representations ",
                repr(self.superrep),
                " and ",
                repr(other.superrep),
            ]))
        if (
            (self.isbra and other.isket)
            or (self.isoperbra and other.isoperket)
        ):
            return _data.inner_csr(self.data, other.data)
        try:
            type = _MATMUL_TYPE_LOOKUP[(self.type, other.type)]
        except KeyError:
            raise TypeError(
                "incompatible matmul types "
                + repr(self.type) + " and " + repr(other.type)
            ) from None
        return Qobj(_data.matmul_csr(self.data, other.data),
                    dims=[self.dims[0], other.dims[1]],
                    type=type,
                    isunitary=self._isunitary and other._isunitary,
                    superrep=self.superrep,
                    copy=False)

    @_tidyup
    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self.__mul__(1 / complex(other))

    @_tidyup
    def __neg__(self):
        return Qobj(_data.neg_csr(self._data),
                    dims=self.dims.copy(),
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def __getitem__(self, ind):
        out = self._data.as_scipy()[ind]
        return out.toarray() if scipy.sparse.issparse(out) else out

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Qobj) or self.dims != other.dims:
            return False
        diff = _data.sub_csr(self._data, other._data)
        return np.all(np.abs(diff.as_scipy().data) < settings.atol)

    @_tidyup
    def __pow__(self, n, m=None):  # calculates powers of Qobj
        if (
            self.type not in ('oper', 'super')
            or self.dims[0] != self.dims[1]
            or m is not None
            or not isinstance(n, numbers.Integral)
            or n < 0
        ):
            return NotImplemented
        return Qobj(_data.pow_csr(self._data, n),
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def _str_header(self):
        out = ", ".join([
            "Quantum object: dims=" + str(self.dims),
            "shape=" + str(self.data.shape),
            "type=" + repr(self.type),
        ])
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
            data = self.data.as_scipy()
        elif all(np.imag(self.data.as_scipy().data) == 0):
            data = np.real(self.full())
        else:
            data = self.full()
        return self._str_header() + "\nQobj data =\n" + str(data)

    def __repr__(self):
        # give complete information on Qobj without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    def __call__(self, other):
        """
        Acts this Qobj on another Qobj either by left-multiplication,
        or by vectorization and devectorization, as
        appropriate.
        """
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")
        if (self.type, other.type) not in _CALL_ALLOWED:
            raise TypeError(self.type + " cannot act on " + other.type)
        if self.issuper:
            if other.isket:
                other = other.proj()
            return vector_to_operator(self @ operator_to_vector(other))
        return self.__matmul__(other)

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
        data = r'\begin{equation*}\left(\begin{array}{*{11}c}'
        data += r"\\".join(_latex_row(row, cols, self.data.as_scipy())
                           for row in rows)
        data += r'\end{array}\right)\end{equation*}'
        return self._str_header() + data

    def dag(self):
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        return Qobj(_data.adjoint_csr(self._data),
                    dims=[self.dims[1], self.dims[0]],
                    type=_ADJOINT_TYPE_LOOKUP[self.type],
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def conj(self):
        """Get the element-wise conjugation of the quantum object."""
        return Qobj(_data.conj_csr(self._data),
                    dims=self.dims.copy(),
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def trans(self):
        """Get the matrix transpose of the quantum operator.

        Returns
        -------
        oper : :class:`.Qobj`
            Transpose of input operator.
        """
        return Qobj(_data.transpose_csr(self._data),
                    dims=[self.dims[1], self.dims[0]],
                    type=_ADJOINT_TYPE_LOOKUP[self.type],
                    superrep=self.superrep,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def dual_chan(self):
        """Dual channel of quantum object representing a completely positive
        map.
        """
        # Uses the technique of Johnston and Kribs (arXiv:1102.0948), which
        # is only valid for completely positive maps.
        if not self.iscp:
            raise ValueError("Dual channels are only implemented for CP maps.")
        J = to_choi(self)
        tensor_idxs = enumerate_flat(J.dims)
        J_dual = tensor_swap(J, *(
                list(zip(tensor_idxs[0][1], tensor_idxs[0][0])) +
                list(zip(tensor_idxs[1][1], tensor_idxs[1][0]))
        )).trans()
        J_dual.superrep = 'choi'
        return J_dual

    def norm(self, norm=None, kwargs=None):
        """
        Norm of a quantum object.

        Default norm is L2-norm for kets and trace-norm for operators.  Other
        ket and operator norms may be specified using the `norm` parameter.

        Parameters
        ----------
        norm : str
            Which type of norm to use.  Allowed values for vectors are 'l2' and
            'max'.  Allowed values for matrices are 'tr' for the trace norm,
            'fro' for the Frobenius norm, 'one' and 'max'.

        kwargs : dict
            Additional keyword arguments to pass on to the relevant norm
            solver.  See details for each norm function in :mod:`.data.norm`.

        Returns
        -------
        norm : float
            The requested norm of the operator or state quantum object.
        """
        if self.type in ('oper', 'super'):
            norm = norm or 'tr'
            if norm not in _NORM_ALLOWED_MATRIX:
                raise ValueError(
                    "matrix norm must be in " + repr(_NORM_ALLOWED_MATRIX)
                )
        else:
            norm = norm or 'l2'
            if norm not in _NORM_ALLOWED_VECTOR:
                raise ValueError(
                    "vector norm must be in " + repr(_NORM_ALLOWED_VECTOR)
                )
        kwargs = kwargs or {}
        return _NORM_FUNCTION_LOOKUP[norm](self.data, **kwargs)

    def proj(self):
        """Form the projector from a given ket or bra vector.

        Parameters
        ----------
        Q : :class:`qutip.Qobj`
            Input bra or ket vector

        Returns
        -------
        P : :class:`qutip.Qobj`
            Projection operator.
        """
        if not (self.isket or self.isbra):
            raise TypeError("projection is only defined for bras and kets")
        dims = ([self.dims[0], self.dims[0]] if self.isket
                else [self.dims[1], self.dims[1]])
        return Qobj(_data.project_csr(self._data),
                    dims=dims,
                    type='oper',
                    isherm=True,
                    copy=False)

    def tr(self):
        """Trace of a quantum object.

        Returns
        -------
        trace : float
            Returns the trace of the quantum object.

        """
        out = _data.trace_csr(self._data)
        return out.real if self.isherm else out

    def purity(self):
        """Calculate purity of a quantum object.

        Returns
        -------
        state_purity : float
            Returns the purity of a quantum object.
            For a pure state, the purity is 1.
            For a mixed state of dimension `d`, 1/d<=purity<1.

        """
        if self.type in ("super", "operator-ket", "operator-bra"):
            raise TypeError('purity is only defined for states.')
        if self.isket or self.isbra:
            return _data.norm.l2_csr(self.data)**2
        return _data.trace_csr(self.data @ self.data).real

    def full(self, order='C', squeeze=False):
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

    def diag(self):
        """Diagonal elements of quantum object.

        Returns
        -------
        diags : array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.
        """
        out = self.data.as_scipy().diagonal()
        if np.any(np.imag(out) > settings.atol) or not self.isherm:
            return out
        else:
            return np.real(out)

    @_tidyup
    def expm(self, method='dense'):
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Parameters
        ----------
        method : str {'dense', 'sparse'}
            Use set method to use to calculate the matrix exponentiation. The
            available choices includes 'dense' and 'sparse'.  Since the
            exponential of a matrix is nearly always dense, method='dense'
            is set as default.s

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.

        """
        if self.dims[0] != self.dims[1]:
            raise TypeError("expm is only valid for square operators")
        if method == 'dense':
            # TODO: swap back to the proper output once the dispatcher is
            # implemented.
            # data = _data.expm_csr_dense(self.data)
            data = _data.create(_data.expm_csr_dense(self.data).to_array())
        elif method == 'sparse':
            data = _data.expm_csr(self.data)
        else:
            raise ValueError("method must be 'dense' or 'sparse'")
        return Qobj(data,
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    isherm=self._isherm,
                    copy=False)

    def check_herm(self):
        """Check if the quantum object is hermitian.

        Returns
        -------
        isherm : bool
            Returns the new value of isherm property.
        """
        self._isherm = None
        return self.isherm

    def sqrtm(self, sparse=False, tol=0, maxiter=100000):
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
        oper : :class:`qutip.Qobj`
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
        if self.dims[0] != self.dims[1]:
            raise TypeError('sqrt only valid on square matrices')
        evals, evecs = _data.eigs_csr(self.data, self.isherm,
                                      sparse=sparse, tol=tol,
                                      maxiter=maxiter)
        evecs = np.hstack(evecs)
        numevals = len(evals)
        dV = scipy.sparse.spdiags(np.sqrt(evals, dtype=complex), 0,
                                  numevals, numevals,
                                  format='csr')
        if self.isherm:
            spDv = dV.dot(evecs.conj().T)
        else:
            spDv = dV.dot(np.linalg.inv(evecs))
        return Qobj(evecs.dot(spDv),
                    dims=self.dims,
                    type=self.type,
                    superrep=self.superrep,
                    copy=False)

    def cosm(self):
        """Cosine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix cosine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.

        """
        if self.dims[0] != self.dims[1]:
            raise TypeError('invalid operand for matrix cosine')
        return 0.5 * ((1j * self).expm() + (-1j * self).expm())

    def sinm(self):
        """Sine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix sine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.
        """
        if self.dims[0] != self.dims[1]:
            raise TypeError('invalid operand for matrix sine')
        return -0.5j * ((1j * self).expm() - (-1j * self).expm())

    def inv(self, sparse=False):
        """Matrix inverse of a quantum operator

        Operator must be square.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Matrix inverse of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.
        """
        if self.data.shape[0] != self.data.shape[1]:
            raise TypeError('Invalid operand for matrix inverse')
        if sparse:
            inv_mat = scipy.sparse.linalg.inv(self.data.as_scipy().tocsc())
        else:
            inv_mat = np.linalg.inv(self.full())
        return Qobj(inv_mat,
                    dims=[self.dims[1], self.dims[0]],
                    type=self.type,
                    superrep=self.superrep,
                    copy=False)

    @_tidyup
    def unit(self, inplace=False, norm=None, kwargs=None):
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
        obj : :class:`qutip.Qobj`
            Normalized quantum object.  Will be the `self` object if in place.
        """
        norm = self.norm(norm=norm, kwargs=kwargs)
        if inplace:
            self.data /= norm
            self._isherm = self._isherm if norm.imag == 0 else None
            self._isunitary = (self._isunitary if abs(norm) - 1 < settings.atol
                               else None)
            out = self
        else:
            out = self / norm
        return out

    @_tidyup
    def ptrace(self, sel):
        """
        Partial trace of the quantum object.

        Parameters
        ----------
        sel : int/list
            An ``int`` or ``list`` of components to keep after partial trace.

        Returns
        -------
        oper : :class:`qutip.Qobj`
            Quantum object representing partial trace with selected components
            remaining.
        """
        # TODO: reorganise ptrace functions into proper data layer bits.
        try:
            sel = sorted(sel)
        except TypeError:
            if not isinstance(sel, numbers.Integral):
                raise TypeError(
                    "selection must be an integer or list of integers"
                ) from None
            sel = [sel]
        data, dims = _data.ptrace_csr(self, sel)
        # TODO: how is the partial trace of a superoperator defined?  Why is it
        # of type 'oper' not 'super'?
        return Qobj(data,
                    dims=dims,
                    type='oper',
                    copy=False)

    def contract(self, inplace=False):
        """
        Contract subspaces of the tensor structure which are 1D.  Not defined
        on superoperators.  If all dimensions are scalar, a Qobj of dimension
        [[1], [1]] is returned, i.e. _multiple_ scalar dimensions are
        contracted, but one is left.

        Parameters
        ----------
        inplace: bool, optional
            If True, modify the dimensions in place.  If False, return a copied
            object.

        Returns
        -------
        out: :class:`.Qobj`
            Quantum object with dimensions contracted.  Will be `self` if
            :param:`inplace` is True.
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
        return Qobj(self.data.copy(), dims=dims, type=self.type, copy=False)

    def permute(self, order):
        """
        Permute the tensor structure of a quantum object.  For example,
            qutip.tensor(x, y).permute([1, 0])
        will give the same result as
            qutip.tensor(y, x)
        and
            qutip.tensor(a, b, c).permute([1, 2, 0])
        will be the same as
            qutip.tensor(b, c, a)

        For regular objects (bras, kets and operators) we expect `order` to be
        a flat list of integers, which specifies the new order of the tensor
        product.

        For superoperators, we expect `order` to be something like
            [[0, 2], [1, 3]]
        which tells us to permute according to [0, 2, 1, 3], and then group
        indices according to the length of each sublist.  As another example,
        permuting a superoperator with dimensions of
          [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
        by an `order`
          [[0, 3], [1, 4], [2, 5]]
        should give a new object with dimensions
          [[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]].

        Parameters
        ----------
        order : list
            List of indices specifying the new tensor order.

        Returns
        -------
        P : :class:`qutip.Qobj`
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
                if self.dims[0] != self.dims[1]:
                    raise TypeError("undefined for non-square operators")
                dims = [new_structure, new_structure]
            data = _data.permute.dimensions_csr(self.data, structure, order)
            return Qobj(data,
                        dims=dims,
                        type=self.type,
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
            if self.dims[0] != self.dims[1]:
                raise TypeError("undefined for non-square operators")
            dims = [new_structure, new_structure]
        data = _data.permute.dimensions_csr(self.data,
                                            flat_structure, flat_order)
        return Qobj(data,
                    dims=dims,
                    type=self.type,
                    superrep=self.superrep,
                    copy=False)

    def tidyup(self, atol=None):
        """
        Removes small elements from the quantum object.

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
        atol = atol or settings.auto_tidyup_atol
        self.data = _data.tidyup_csr(self.data, atol)
        return self

    def transform(self, inpt, inverse=False):
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
        oper : :class:`qutip.Qobj`
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
            sci = scipy.sparse.hstack([psi.data.as_scipy() for psi in inpt],
                                      format='csr', dtype=np.complex128)
            S = _data.create(sci).adjoint()
        elif isinstance(inpt, Qobj) and inpt.isoper:
            S = inpt.data
        elif isinstance(inpt, np.ndarray):
            S = _data.create(inpt).conj()
        else:
            raise TypeError('Invalid operand for basis transformation')

        # transform data
        if inverse:
            if self.isket:
                data = S.adjoint() @ self.data
            elif self.isbra:
                data = self.data @ S
            else:
                data = S.adjoint() @ self.data @ S
        else:
            if self.isket:
                data = S @ self.data
            elif self.isbra:
                data = self.data @ S.adjoint()
            else:
                data = S @ self.data @ S.adjoint()
        return Qobj(data,
                    dims=self.dims,
                    type=self.type,
                    isherm=self._isherm,
                    superrep=self.superrep,
                    copy=False)

    def trunc_neg(self, method="clip"):
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
        oper : :class:`qutip.Qobj`
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
        out_data = _data.csr.zeros(*self.shape)
        for value, state in zip(eigvals, eigstates):
            if value:
                # add in 3-argument form is fused-add-multiply
                out_data = _data.add_csr(out_data,
                                         _data.project_csr(state.data), value)
        out_data /= _data.norm.trace_csr(out_data)
        return Qobj(out_data,
                    dims=self.dims.copy(),
                    type=self.type,
                    isherm=True,
                    copy=False)

    def matrix_element(self, bra, ket):
        """Calculates a matrix element.

        Gives the matrix element for the quantum object sandwiched between a
        `bra` and `ket` vector.

        Parameters
        -----------
        bra : :class:`qutip.Qobj`
            Quantum object of type 'bra' or 'ket'

        ket : :class:`qutip.Qobj`
            Quantum object of type 'ket'.

        Returns
        -------
        elem : complex
            Complex valued matrix element.

        Note
        ----
        It is slightly more computationally efficient to use a ket
        vector for the 'bra' input.

        """
        if not self.isoper:
            raise TypeError("Can only get matrix elements for an operator.")
        if bra.type not in ('bra', 'ket') or ket.type not in ('bra', 'ket'):
            msg = "Can only calculate matrix elements between a bra and a ket."
            raise TypeError(msg)
        left, op, right = bra.data, self.data, ket.data
        if ket.isbra:
            right = right.adjoint()
        return _data.inner_op_csr(left, op, right, bra.isket)

    def overlap(self, other):
        """
        Overlap between two state vectors or two operators.

        Gives the overlap (inner product) between the current bra or ket Qobj
        and and another bra or ket Qobj. It gives the Hilbert-Schmidt overlap
        when one of the Qobj is an operator/density matrix.

        Parameters
        -----------
        other : :class:`qutip.Qobj`
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
                left = _data.project_csr(left)
            if not other.isoper:
                right = _data.project_csr(right)
            return _data.trace_csr(left @ right)
        if other.isbra:
            right = right.adjoint()
        out = _data.inner_csr(left, right, self.isket)
        if self.isket and other.isbra:
            # In this particular case, we've basically doing
            #   conj(other.overlap(self))
            # so we take care to conjugate the output.
            out = np.conj(out)
        return out

    def eigenstates(self, sparse=False, sort='low', eigvals=0,
                    tol=0, maxiter=100000, phase_fix=None):
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
        evals, evecs = _data.eigs_csr(self.data, self.isherm, sparse=sparse,
                                      sort=sort, eigvals=eigvals, tol=tol,
                                      maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        ekets = np.array([Qobj(vec, dims=new_dims, copy=False)
                          for vec in evecs],
                         dtype=object)
        norms = np.array([ket.norm() for ket in ekets])
        if phase_fix is None:
            phase = np.array([1] * len(ekets))
        else:
            phase = np.array([np.abs(ket[phase_fix, 0]) / ket[phase_fix, 0]
                              if ket[phase_fix, 0] else 1
                              for ket in ekets])
        return evals, ekets / norms * phase

    def eigenenergies(self, sparse=False, sort='low',
                      eigvals=0, tol=0, maxiter=100000):
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
        return _data.eigs_csr(self.data, self.isherm, vecs=False,
                              sparse=sparse, sort=sort, eigvals=eigvals,
                              tol=tol, maxiter=maxiter)

    def groundstate(self, sparse=False, tol=0, maxiter=100000, safe=True):
        """Ground state Eigenvalue and Eigenvector.

        Defined for quantum operators or superoperators only.

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
        eigvec : :class:`qutip.Qobj`
            Eigenket for the ground state of quantum operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        if safe:
            evals = 2
        else:
            evals = 1
        grndval, grndvec = _data.eigs_csr(self.data, self.isherm,
                                          sparse=sparse, eigvals=evals,
                                          tol=tol, maxiter=maxiter)
        if safe:
            tol = tol or settings.atol
            if (grndval[1]-grndval[0]) <= 10*tol:
                print("WARNING: Ground state may be degenerate. "
                        "Use Q.eigenstates()")
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        grndvec = Qobj(grndvec[0], dims=new_dims)
        grndvec = grndvec / grndvec.norm()
        return grndval[0], grndvec

    def dnorm(self, B=None):
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`qutip.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return mts.dnorm(self, B)

    @property
    def ishp(self):
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type in ["super", "oper"]:
            try:
                J = to_choi(self)
                return J.isherm
            except:
                return False
        else:
            return False

    @property
    def iscp(self):
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type not in ["super", "oper"]:
            return False
        # We can test with either Choi or chi, since the basis
        # transformation between them is unitary and hence preserves
        # the CP and TP conditions.
        J = self if self.superrep in ('choi', 'chi') else to_choi(self)
        # If J isn't hermitian, then that could indicate either that J is not
        # normal, or is normal, but has complex eigenvalues.  In either case,
        # it makes no sense to then demand that the eigenvalues be
        # non-negative.
        return J.isherm and np.all(J.eigenenergies() >= -settings.atol)

    @property
    def istp(self):
        if self.type not in ['super', 'oper']:
            return False
        # Normalize to a super of type choi or chi.
        # We can test with either Choi or chi, since the basis
        # transformation between them is unitary and hence
        # preserves the CP and TP conditions.
        if self.issuper and self.superrep in ('choi', 'chi'):
            qobj = self
        else:
            qobj = to_choi(self)
        # Possibly collapse dims.
        if any([len(index) > 1
                for super_index in qobj.dims
                for index in super_index]):
            qobj = Qobj(qobj.data,
                        dims=collapse_dims_super(qobj.dims),
                        type=qobj.type,
                        superrep=qobj.superrep,
                        copy=False)
        # We use the condition from John Watrous' lecture notes,
        # Tr_1(J(Phi)) = identity_2.
        tr_oper = qobj.ptrace([0])
        return np.allclose(tr_oper.full(), np.eye(tr_oper.shape[0]),
                           atol=settings.atol)

    @property
    def iscptp(self):
        if not (self.issuper or self.isoper):
            return False
        reps = ('choi', 'chi')
        q_oper = to_choi(self) if self.superrep not in reps else self
        return q_oper.iscp and q_oper.istp

    @property
    def isherm(self):
        if self._isherm is not None:
            return self._isherm
        self._isherm = _data.isherm_csr(self._data)
        return self._isherm

    @isherm.setter
    def isherm(self, isherm):
        self._isherm = isherm

    def _calculate_isunitary(self):
        """
        Checks whether qobj is a unitary matrix
        """
        if not self.isoper or self._data.shape[0] != self._data.shape[1]:
            return False
        iden = _data.csr.identity(self.shape[0])
        cmp = self._data @ self._data.adjoint()
        return np.all(np.abs((cmp - iden).as_scipy().data) < settings.atol)

    @property
    def isunitary(self):
        if self._isunitary is not None:
            return self._isunitary
        self._isunitary = self._calculate_isunitary()
        return self._isunitary

    @property
    def shape(self): return self.data.shape

    isbra = property(isbra)
    isket = property(isket)
    isoper = property(isoper)
    issuper = property(issuper)
    isoperbra = property(isoperbra)
    isoperket = property(isoperket)


def ptrace(Q, sel):
    """
    Partial trace of the Qobj with selected components remaining.

    Parameters
    ----------
    Q : :class:`qutip.Qobj`
        Composite quantum object.
    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    Returns
    -------
    oper : :class:`qutip.Qobj`
        Quantum object representing partial trace with selected components
        remaining.

    Notes
    -----
    This function is for legacy compatibility only. It is recommended to use
    the ``ptrace()`` Qobj method.
    """
    if not isinstance(Q, Qobj):
        raise TypeError("Input is not a quantum object")
    return Q.ptrace(sel)


# TRAILING IMPORTS
# We do a few imports here to avoid circular dependencies.
from qutip.core.superop_reps import to_choi
from qutip.core.superoperator import vector_to_operator, operator_to_vector
from qutip.core.tensor import tensor_swap
from qutip.core import metrics as mts
