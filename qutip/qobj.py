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
"""Main module for QuTiP, consisting of the Quantum Object (Qobj) class and
its methods.
"""
import warnings
import types
import pickle

try:
    import builtins
except:
    import __builtin__ as builtins

# import math functions from numpy.math: required for td string evaluation
from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   ceil, copysign, cos, cosh, degrees, e, exp, expm1, fabs,
                   floor, fmod, frexp, hypot, isinf, isnan, ldexp, log, log10,
                   log1p, modf, pi, radians, sin, sinh, sqrt, tan, tanh, trunc)

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import qutip.settings as settings
from qutip import __version__
from qutip.ptrace import _ptrace
from qutip.permute import _permute
from qutip.sparse import (sp_eigs, sp_expm, sp_fro_norm, sp_max_norm,
                          sp_one_norm, sp_L2_norm, sp_inf_norm)


class Qobj(object):
    """A class for representing quantum objects, such as quantum operators
    and states.

    The Qobj class is the QuTiP representation of quantum operators and state
    vectors. This class also implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator/state operations.  The Qobj constructor optionally takes a
    dimension ``list`` and/or shape ``list`` as arguments.

    Parameters
    ----------
    inpt : array_like
        Data for vector/matrix representation of the quantum object.
    dims : list
        Dimensions of object used for tensor products.
    shape : list
        Shape of underlying data structure (matrix shape).
    fast : bool
        Flag for fast qobj creation when running ode solvers.
        This parameter is used internally only.


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
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
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
    conj()
        Conjugate of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    eigenenergies(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies (eigenvalues) of a quantum object.
    eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies and eigenstates of quantum object.
    expm()
        Matrix exponential of quantum object.
    full()
        Returns dense array of quantum object `data` attribute.
    groundstate(sparse=False,tol=0,maxiter=100000)
        Returns eigenvalue and eigenket for the groundstate of a quantum
        object.
    matrix_element(bra, ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns norm of a ket or an operator.
    permute(order)
        Returns composite qobj with indices reordered.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
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
    unit(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns normalized quantum object.
    """
    __array_priority__ = 100  # sets Qobj priority above numpy arrays

    def __init__(self, inpt=None, dims=[[], []], shape=[],
                 type=None, isherm=None, fast=False, superrep=None):
        """
        Qobj constructor.
        """
        self._isherm = None
        self._type = None
        self.superrep = None

        if fast == 'mc':
            # fast Qobj construction for use in mcsolve with ket output
            self.data = sp.csr_matrix(inpt, dtype=complex)
            self.dims = dims
            self._isherm = False
            return

        if fast == 'mc-dm':
            # fast Qobj construction for use in mcsolve with dm output
            self.data = sp.csr_matrix(inpt, dtype=complex)
            self.dims = dims
            self._isherm = True
            return

        if isinstance(inpt, Qobj):
            # if input is already Qobj then return identical copy

            # make sure matrix is sparse (safety check)
            self.data = sp.csr_matrix(inpt.data, dtype=complex)

            if not np.any(dims):
                # Dimensions of quantum object used for keeping track of tensor
                # components
                self.dims = inpt.dims
            else:
                self.dims = dims

            self.superrep = inpt.superrep

        elif inpt is None:
            # initialize an empty Qobj with correct dimensions and shape

            if any(dims):
                N, M = np.prod(dims[0]), np.prod(dims[1])
                self.dims = dims

            elif shape:
                N, M = shape
                self.dims = [[N], [M]]

            else:
                N, M = 1, 1
                self.dims = [[N], [M]]

            self.data = sp.csr_matrix((N, M), dtype=complex)

        elif isinstance(inpt, list) or isinstance(inpt, tuple):
            # case where input is a list
            if len(np.array(inpt).shape) == 1:
                # if list has only one dimension (i.e [5,4])
                inpt = np.array([inpt]).transpose()
            else:  # if list has two dimensions (i.e [[5,4]])
                inpt = np.array(inpt)

            self.data = sp.csr_matrix(inpt, dtype=complex)

            if not np.any(dims):
                self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
            else:
                self.dims = dims

        elif isinstance(inpt, np.ndarray) or sp.issparse(inpt):
            # case where input is array or sparse
            if inpt.ndim == 1:
                inpt = inpt[:, np.newaxis]

            self.data = sp.csr_matrix(inpt, dtype=complex)

            if not np.any(dims):
                self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
            else:
                self.dims = dims

        elif isinstance(inpt, (int, float, complex,
                               np.integer, np.floating, np.complexfloating)):
            # if input is int, float, or complex then convert to array
            self.data = sp.csr_matrix([[inpt]], dtype=complex)

            if not np.any(dims):
                self.dims = [[1], [1]]
            else:
                self.dims = dims

        else:
            warnings.warn("Initializing Qobj from unsupported type: %s" %
                          builtins.type(inpt))
            inpt = np.array([[0]])
            self.data = sp.csr_matrix(inpt, dtype=complex)
            self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]

        if type == 'super':
            if self.type == 'oper':
                self.dims = [[[d] for d in self.dims[0]],
                             [[d] for d in self.dims[1]]]

        if superrep:
            self.superrep = superrep
        else:
            if self.type == 'super' and self.superrep is None:
                self.superrep = 'super'

        # clear type cache
        self._type = None

    def __add__(self, other):
        """
        ADDITION with Qobj on LEFT [ ex. Qobj+4 ]
        """
        if isinstance(other, eseries):
            return other.__radd__(self)

        if not isinstance(other, Qobj):
            other = Qobj(other)

        if np.prod(other.shape) == 1 and np.prod(self.shape) != 1:
            # case for scalar quantum object
            dat = other.data[0, 0]
            if dat == 0:
                return self

            out = Qobj()

            if self.type in ['oper', 'super']:
                out.data = self.data + dat * sp.identity(
                    self.shape[0], dtype=complex, format='csr')
            else:
                out.data = self.data
                out.data.data = out.data.data + dat

            out.dims = self.dims
            if isinstance(dat, (int, float)):
                out._isherm = self._isherm
            else:
                out._isherm = out.isherm

            out.superrep = self.superrep

            return out.tidyup() if settings.auto_tidyup else out

        elif np.prod(self.shape) == 1 and np.prod(other.shape) != 1:
            # case for scalar quantum object
            dat = self.data[0, 0]
            if dat == 0:
                return other

            out = Qobj()
            if other.type in ['oper', 'super']:
                out.data = dat * sp.identity(other.shape[0], dtype=complex,
                                             format='csr') + other.data
            else:
                out.data = other.data
                out.data.data = out.data.data + dat
            out.dims = other.dims

            if isinstance(dat, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            out.superrep = self.superrep

            return out.tidyup() if settings.auto_tidyup else out

        elif self.dims != other.dims:
            raise TypeError('Incompatible quantum object dimensions')

        elif self.shape != other.shape:
            raise TypeError('Matrix shapes do not match')

        else:  # case for matching quantum objects
            out = Qobj()
            out.data = self.data + other.data
            out.dims = self.dims

            if self.type in ['ket', 'bra', 'operator-ket', 'operator-bra']:
                out._isherm = False
            elif self._isherm is None or other._isherm is None:
                out._isherm = out.isherm
            elif not self._isherm and not other._isherm:
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm and other._isherm

            if self.superrep and other.superrep:
                if self.superrep != other.superrep:
                    msg = ("Adding superoperators with different " +
                           "representations")
                    warnings.warn(msg)

                out.superrep = self.superrep

            return out.tidyup() if settings.auto_tidyup else out

    def __radd__(self, other):
        """
        ADDITION with Qobj on RIGHT [ ex. 4+Qobj ]
        """
        return self + other

    def __sub__(self, other):
        """
        SUBTRACTION with Qobj on LEFT [ ex. Qobj-4 ]
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        SUBTRACTION with Qobj on RIGHT [ ex. 4-Qobj ]
        """
        return (-self) + other

    def __mul__(self, other):
        """
        MULTIPLICATION with Qobj on LEFT [ ex. Qobj*4 ]
        """
        if isinstance(other, Qobj):
            if self.dims[1] == other.dims[0]:
                out = Qobj()
                out.data = self.data * other.data
                dims = [self.dims[0], other.dims[1]]
                out.dims = dims
                if (not isinstance(dims[0][0], list) and
                        not isinstance(dims[1][0], list)):
                    r = range(len(dims[0]))
                    mask = [dims[0][n] == dims[1][n] == 1 for n in r]
                    out.dims = [max([1], [dims[0][n]
                                          for n in r if not mask[n]]),
                                max([1], [dims[1][n]
                                          for n in r if not mask[n]])]
                else:
                    out.dims = dims

                out._isherm = out.isherm

                if self.superrep and other.superrep:
                    if self.superrep != other.superrep:
                        msg = ("Multiplying superoperators with different " +
                               "representations")
                        warnings.warn(msg)

                    out.superrep = self.superrep

                return out.tidyup() if settings.auto_tidyup else out

            elif np.prod(self.shape) == 1:
                out = Qobj(other)
                out.data *= self.data[0, 0]
                out.superrep = other.superrep
                return out.tidyup() if settings.auto_tidyup else out

            elif np.prod(other.shape) == 1:
                out = Qobj(self)
                out.data *= other.data[0, 0]
                out.superrep = self.superrep
                return out.tidyup() if settings.auto_tidyup else out

            else:
                raise TypeError("Incompatible Qobj shapes")

        elif isinstance(other, (list, np.ndarray)):
            # if other is a list, do element-wise multiplication
            return np.array([self * item for item in other])

        elif isinstance(other, eseries):
            return other.__rmul__(self)

        elif isinstance(other, (int, float, complex,
                                np.integer, np.floating, np.complexfloating)):
            out = Qobj()
            out.data = self.data * other
            out.dims = self.dims
            out.superrep = self.superrep
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            return out.tidyup() if settings.auto_tidyup else out

        else:
            raise TypeError("Incompatible object for multiplication")

    def __rmul__(self, other):
        """
        MULTIPLICATION with Qobj on RIGHT [ ex. 4*Qobj ]
        """

        if isinstance(other, (list, np.ndarray)):
            # if other is a list, do element-wise multiplication
            return np.array([item * self for item in other])

        if isinstance(other, eseries):
            return other.__mul__(self)

        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            out = Qobj()
            out.data = other * self.data
            out.dims = self.dims
            out.superrep = self.superrep
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            return out.tidyup() if settings.auto_tidyup else out

        else:
            raise TypeError("Incompatible object for multiplication")

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        """
        DIVISION (by numbers only)
        """
        if isinstance(other, Qobj):  # if both are quantum objects
            raise TypeError("Incompatible Qobj shapes " +
                            "[division with Qobj not implemented]")

        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            out = Qobj()
            out.data = self.data / other
            out.dims = self.dims
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            out.superrep = self.superrep

            return out.tidyup() if settings.auto_tidyup else out

        else:
            raise TypeError("Incompatible object for division")

    def __neg__(self):
        """
        NEGATION operation.
        """
        out = Qobj()
        out.data = -self.data
        out.dims = self.dims
        out.superrep = self.superrep
        out._isherm = self._isherm
        return out.tidyup() if settings.auto_tidyup else out

    def __getitem__(self, ind):
        """
        GET qobj elements.
        """
        out = self.data[ind]
        if sp.issparse(out):
            return np.asarray(out.todense())
        else:
            return out

    def __eq__(self, other):
        """
        EQUALITY operator.
        """
        if (isinstance(other, Qobj) and
                self.dims == other.dims and
                not np.any(np.abs((self.data - other.data).data) >
                           settings.atol)):
            return True
        else:
            return False

    def __ne__(self, other):
        """
        INEQUALITY operator.
        """
        return not (self == other)

    def __pow__(self, n, m=None):  # calculates powers of Qobj
        """
        POWER operation.
        """
        if self.type not in ['oper', 'super']:
            raise Exception("Raising a qobj to some power works only for " +
                            "operators and super-operators (square matrices).")

        if m is not None:
            raise NotImplementedError("modulo is not implemented for Qobj")

        try:
            data = self.data ** n
            out = Qobj(data, dims=self.dims)
            out.superrep = self.superrep
            return out.tidyup() if settings.auto_tidyup else out

        except:
            raise ValueError('Invalid choice of exponent.')

    def __abs__(self):
        return abs(self.data)

    def __str__(self):
        s = ""
        t = self.type
        shape = self.shape
        if self.type in ['oper', 'super']:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t +
                  ", isherm = " + str(self.isherm) +
                  (
                      ", superrep = {0.superrep}".format(self)
                      if t == "super" and self.superrep != "super"
                      else ""
                  ) + "\n")
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t + "\n")
        s += "Qobj data =\n"

        if shape[0] > 10000 or shape[1] > 10000:
            # if the system is huge, don't attempt to convert to a
            # dense matrix and then to string, because it is pointless
            # and is likely going to produce memory errors. Instead print the
            # sparse data string representation
            s += str(self.data)

        elif all(np.imag(self.data.data) == 0):
            s += str(np.real(self.full()))

        else:
            s += str(self.full())

        return s

    def __repr__(self):
        # give complete information on Qobj without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

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
        t = self.type
        shape = self.shape
        s = r''
        if self.type in ['oper', 'super']:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t +
                  ", isherm = " + str(self.isherm) +
                  (
                      ", superrep = {0.superrep}".format(self)
                      if t == "super" and self.superrep != "super"
                      else ""
                  ))
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t)

        M, N = self.data.shape

        s += r'\begin{equation*}\left(\begin{array}{*{11}c}'

        def _format_float(value):
            if value == 0.0:
                return "0.0"
            elif abs(value) > 1000.0 or abs(value) < 0.001:
                return ("%.3e" % value).replace("e", r"\times10^{") + "}"
            elif abs(value - int(value)) < 0.001:
                return "%.1f" % value
            else:
                return "%.3f" % value

        def _format_element(m, n, d):
            s = " & " if n > 0 else ""
            if type(d) == str:
                return s + d
            else:
                if abs(np.imag(d)) < settings.atol:
                    return s + _format_float(np.real(d))
                elif abs(np.real(d)) < settings.atol:
                    return s + _format_float(np.imag(d)) + "j"
                else:
                    s_re = _format_float(np.real(d))
                    s_im = _format_float(np.imag(d))
                    if np.imag(d) > 0.0:
                        return (s + "(" + s_re + "+" + s_im + "j)")
                    else:
                        return (s + "(" + s_re + s_im + "j)")

        if M > 10 and N > 10:
            # truncated matrix output
            for m in range(5):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

            for n in range(5):
                s += _format_element(m, n, r'\vdots')
            s += r' & \ddots'
            for n in range(N - 5, N):
                s += _format_element(m, n, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        elif M > 10 and N == 1:
            # truncated column vector output
            for m in range(5):
                s += _format_element(m, 0, self.data[m, 0])
                s += r'\\'

            s += _format_element(m, 0, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                s += _format_element(m, 0, self.data[m, 0])
                s += r'\\'

        elif M == 1 and N > 10:
            # truncated row vector output
            for n in range(5):
                s += _format_element(0, n, self.data[0, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(0, n, self.data[0, n])
            s += r'\\'

        else:
            # full output
            for m in range(M):
                for n in range(N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        s += r'\end{array}\right)\end{equation*}'
        return s

    def dag(self):
        """Adjoint operator of quantum object.
        """
        out = Qobj()
        out.data = self.data.T.conj().tocsr()
        out.dims = [self.dims[1], self.dims[0]]
        out._isherm = self._isherm
        return out

    def conj(self):
        """Conjugate operator of quantum object.
        """
        out = Qobj()
        out.data = self.data.conj()
        out.dims = [self.dims[0], self.dims[1]]
        return out

    def norm(self, norm=None, sparse=False, tol=0, maxiter=100000):
        """Norm of a quantum object.

        Default norm is L2-norm for kets and trace-norm for operators.
        Other ket and operator norms may be specified using the `norm` and
        argument.

        Parameters
        ----------
        norm : str
            Which norm to use for ket/bra vectors: L2 'l2', max norm 'max',
            or for operators: trace 'tr', Frobius 'fro', one 'one', or max
            'max'.

        sparse : bool
            Use sparse eigenvalue solver for trace norm.  Other norms are not
            affected by this parameter.

        tol : float
            Tolerance for sparse solver (if used) for trace norm. The sparse
            solver may not converge if the tolerance is set too low.

        maxiter : int
            Maximum number of iterations performed by sparse solver (if used)
            for trace norm.

        Returns
        -------
        norm : float
            The requested norm of the operator or state quantum object.


        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        if self.type in ['oper', 'super']:
            if norm is None or norm == 'tr':
                vals = sp_eigs(self.data, self.isherm, vecs=False,
                               sparse=sparse, tol=tol, maxiter=maxiter)
                return np.sum(sqrt(abs(vals) ** 2))
            elif norm == 'fro':
                return sp_fro_norm(self.data)
            elif norm == 'one':
                return sp_one_norm(self.data)
            elif norm == 'max':
                return sp_max_norm(self.data)
            else:
                raise ValueError(
                    "For matrices, norm must be 'tr', 'fro', 'one', or 'max'.")
        else:
            if norm is None or norm == 'l2':
                return sp_L2_norm(self.data)
            elif norm == 'max':
                return sp_max_norm(self.data)
            else:
                raise ValueError("For vectors, norm must be 'l2', or 'max'.")

    def tr(self):
        """Trace of a quantum object.

        Returns
        -------
        trace: float
            Returns ``real`` if operator is Hermitian, returns ``complex``
            otherwise.

        """
        if self.isherm:
            return float(np.real(np.sum(self.data.diagonal())))
        else:
            return complex(np.sum(self.data.diagonal()))

    def full(self, squeeze=False):
        """Dense array from quantum object.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.

        """
        if squeeze:
            return self.data.toarray().squeeze()
        else:
            return self.data.toarray()

    def diag(self):
        """Diagonal elements of quantum object.

        Returns
        -------
        diags: array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.

        """
        out = self.data.diagonal()
        if np.any(np.imag(out) > settings.atol) or not self.isherm:
            return out
        else:
            return np.real(out)

    def expm(self, method=None):
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Parameters
        ----------
        method : str {'dense', 'sparse', 'scipy-dense', 'scipy-sparse'}
            Use set method to use to calculate the matrix exponentiation. The
            available choices includes 'dense' and 'sparse' for using QuTiP's
            implementation of expm using dense and sparse matrices,
            respectively, and 'scipy-dense' and 'scipy-sparse' for using the
            scipy.linalg.expm (dense) and scipy.sparse.linalg.expm (sparse).
            If no method is explicitly given a heuristic will be used to try
            and automatically select the most appropriate solver.

        Returns
        -------
        oper : qobj
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.

        """
        if self.dims[0][0] != self.dims[1][0]:
            raise TypeError('Invalid operand for matrix exponential')

        if method == 'dense':
            F = sp_expm(self.data, sparse=False)

        elif method == 'sparse':
            F = sp_expm(self.data, sparse=True)

        elif method == 'scipy-dense':
            F = la.expm(self.full())

        elif method == 'scipy-sparse':
            F = sp.linalg.expm(self.data.tocsc())

        else:
            # if method is not explicitly given, try to make a good choice
            # between sparse and dense solvers by considering the size of the
            # system and the number of non-zero elements.
            N = self.data.shape[0]
            n = self.data.nnz

            if N ** 2 < 100 * n:
                # large number of nonzero elements, revert to dense solver
                F = la.expm(self.full())
            elif N > 400:
                # large system, and quite sparse -> qutips sparse method
                F = sp_expm(self.data, sparse=True)
            else:
                # small system, but quite sparse -> qutips sparse/dense method
                F = sp_expm(self.data, sparse=False)

        out = Qobj(F, dims=self.dims)
        return out.tidyup() if settings.auto_tidyup else out

    def checkherm(self):
        """Check if the quantum object is hermitian.

        Returns
        -------
        isherm: bool
            Returns the new value of isherm property.
        """
        self._isherm = None
        return self.isherm

    def sqrtm(self, sparse=False, tol=0, maxiter=100000):
        """Sqrt of a quantum operator.

        Operator must be square.

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
        oper: qobj
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
        if self.dims[0][0] == self.dims[1][0]:
            evals, evecs = sp_eigs(self.data, self.isherm, sparse=sparse,
                                   tol=tol, maxiter=maxiter)
            numevals = len(evals)
            dV = sp.spdiags(np.sqrt(evals, dtype=complex), 0, numevals,
                            numevals, format='csr')
            if self.isherm:
                spDv = dV.dot(evecs.T.conj().T)
            else:
                spDv = dV.dot(np.linalg.inv(evecs.T))

            out = Qobj(evecs.T.dot(spDv), dims=self.dims)
            return out.tidyup() if settings.auto_tidyup else out

        else:
            raise TypeError('Invalid operand for matrix square root')

    def unit(self, norm=None, sparse=False, tol=0, maxiter=100000):
        """Operator or state normalized to unity.

        Uses norm from Qobj.norm().

        Parameters
        ----------
        norm : str
            Requested norm for states / operators.
        sparse : bool
            Use sparse eigensolver for trace norm. Does not affect other norms.
        tol : float
            Tolerance used by sparse eigensolver.
        maxiter: int
            Number of maximum iterations performed by sparse eigensolver.

        Returns
        -------
        oper : qobj
            Normalized quantum object.

        """
        out = self / self.norm(norm=norm, sparse=sparse,
                               tol=tol, maxiter=maxiter)
        if settings.auto_tidyup:
            return out.tidyup()
        else:
            return out

    def ptrace(self, sel):
        """Partial trace of the quantum object.

        Parameters
        ----------
        sel : int/list
            An ``int`` or ``list`` of components to keep after partial trace.

        Returns
        -------
        oper: qobj
            Quantum object representing partial trace with selected components
            remaining.

        Notes
        -----
        This function is identical to the :func:`qutip.qobj.ptrace` function
        that has been deprecated.

        """
        q = Qobj()
        q.data, q.dims, _ = _ptrace(self, sel)
        return q.tidyup() if settings.auto_tidyup else q

    def permute(self, order):
        """Permutes a composite quantum object.

        Parameters
        ----------
        order : list/array
            List specifying new tensor order.

        Returns
        -------
        P : qobj
            Permuted quantum object.
i
        """
        q = Qobj()
        q.data, q.dims = _permute(self, order)
        return q.tidyup() if settings.auto_tidyup else q

    def tidyup(self, atol=None):
        """Removes small elements from the quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup. Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper: qobj
            Quantum object with small elements removed.

        """
        if atol is None:
            atol = settings.auto_tidyup_atol

        if self.data.nnz:

            data_real = self.data.data.real
            data_real[abs(data_real) < atol] = 0

            data_imag = self.data.data.imag
            data_imag[abs(data_imag) < atol] = 0

            self.data.data = data_real + 1j * data_imag

            self.data.eliminate_zeros()
            return self
        else:
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
        oper : qobj
            Operator in new basis.

        Notes
        -----
        This function is still in development.


        """
        if isinstance(inpt, list) or (isinstance(inpt, np.ndarray) and
                                      len(inpt.shape) == 1):
            if len(inpt) != max(self.shape):
                raise TypeError(
                    'Invalid size of ket list for basis transformation')
            S = np.matrix(np.hstack([psi.full() for psi in inpt])).H
        elif isinstance(inpt, np.ndarray):
            S = np.matrix(inpt)
        elif isinstance(inpt, Qobj) and inpt.isoper:
            S = np.matrix(inpt.full())
        else:
            raise TypeError('Invalid operand for basis transformation')

        # normalize S just in case the supplied basis states aren't normalized
        # S = S/la.norm(S)

        out = Qobj(dims=self.dims)
        out._isherm = self._isherm
        out.superrep = self.superrep

        # transform data
        if inverse:
            if self.isket:
                out.data = S.H * self.data
            elif self.isbra:
                out.data = self.data * S
            else:
                out.data = S.H * self.data * S
        else:
            if self.isket:
                out.data = S * self.data
            elif self.isbra:
                out.data = self.data * S.H
            else:
                out.data = S * self.data * S.H

        # force sparse
        out.data = sp.csr_matrix(out.data, dtype=complex)

        return out

    def matrix_element(self, bra, ket):
        """Calculates a matrix element.

        Gives the matrix element for the quantum object sandwiched between a
        `bra` and `ket` vector.

        Parameters
        -----------
        bra : qobj
            Quantum object of type 'bra'.

        ket : qobj
            Quantum object of type 'ket'.

        Returns
        -------
        elem : complex
            Complex valued matrix element.

        Raises
        ------
        TypeError
            Can only calculate matrix elements between a bra and ket
            quantum object.

        """

        if isinstance(bra, Qobj) and isinstance(ket, Qobj):

            if self.isoper:
                if bra.isbra and ket.isket:
                    return (bra.data * self.data * ket.data)[0, 0]

                if bra.isket and ket.isket:
                    return (bra.data.T * self.data * ket.data)[0, 0]

        raise TypeError("Can only calculate matrix elements for operators " +
                        "and between ket and bra Qobj")

    def overlap(self, state):
        """Overlap between two state vectors.

        Gives the overlap (scalar product) for the quantum object and `state`
        state vector.

        Parameters
        -----------
        state : qobj
            Quantum object for a state vector of type 'ket' or 'bra'.

        Returns
        -------
        overlap : complex
            Complex valued overlap.

        Raises
        ------
        TypeError
            Can only calculate overlap between a bra and ket quantum objects.
        """

        if isinstance(state, Qobj):

            if self.isbra:
                if state.isket:
                    return (self.data * state.data)[0, 0]
                elif state.isbra:
                    return (self.data * state.data.H)[0, 0]

            elif self.isket:
                if state.isbra:
                    return (self.data.H * state.data.H)[0, 0]
                elif state.isket:
                    return (self.data.H * state.data)[0, 0]

        raise TypeError("Can only calculate overlap for state vector Qobjs")

    def eigenstates(self, sparse=False, sort='low',
                    eigvals=0, tol=0, maxiter=100000):
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
        evals, evecs = sp_eigs(self.data, self.isherm, sparse=sparse,
                               sort=sort, eigvals=eigvals, tol=tol,
                               maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        ekets = np.array([Qobj(vec, dims=new_dims) for vec in evecs])
        norms = np.array([ket.norm() for ket in ekets])
        return evals, ekets / norms

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
        eigvals: array
            Array of eigenvalues for operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        return sp_eigs(self.data, self.isherm, vecs=False, sparse=sparse,
                       sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)

    def groundstate(self, sparse=False, tol=0, maxiter=100000):
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

        Returns
        -------
        eigval : float
            Eigenvalue for the ground state of quantum operator.

        eigvec : qobj
            Eigenket for the ground state of quantum operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        grndval, grndvec = sp_eigs(self.data, self.isherm, sparse=sparse,
                                   eigvals=1, tol=tol, maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        grndvec = Qobj(grndvec[0], dims=new_dims)
        grndvec = grndvec / grndvec.norm()
        return grndval[0], grndvec

    def trans(self):
        """Transposed operator.

        Returns
        -------
        oper : qobj
            Transpose of input operator.

        """
        out = Qobj()
        out.data = self.data.T.tocsr()
        out.dims = [self.dims[1], self.dims[0]]
        return out

    def extract_states(self, states_inds, normalize=False):
        """Qobj with states in state_inds only.

        Parameters
        ----------
        states_inds : list of integer
            The states that should be kept.

        normalize : True / False
            Weather or not the new Qobj instance should be normalized (default
            is False). For Qobjs that represents density matrices or state
            vectors normalized should probably be set to True, but for Qobjs
            that represents operators in for example an Hamiltonian, normalize
            should be False.

        Returns
        -------
        q : :class:`qutip.Qobj`

            A new instance of :class:`qutip.Qobj` that contains only the states
            corresponding to the indices in `state_inds`.

        .. note::

            Experimental.

        """
        if self.isoper:
            q = Qobj(self.data[states_inds, :][:, states_inds])
        elif self.isket:
            q = Qobj(self.data[states_inds, :])
        elif self.isbra:
            q = Qobj(self.data[:, states_inds])
        else:
            raise TypeError("Can only eliminate states from operators or " +
                            "state vectors")

        return q.unit() if normalize else q

    def eliminate_states(self, states_inds, normalize=False):
        """Creates a new quantum object with states in state_inds eliminated.

        Parameters
        ----------
        states_inds : list of integer
            The states that should be removed.

        normalize : True / False
            Weather or not the new Qobj instance should be normalized (default
            is False). For Qobjs that represents density matrices or state
            vectors normalized should probably be set to True, but for Qobjs
            that represents operators in for example an Hamiltonian, normalize
            should be False.

        Returns
        -------
        q : :class:`qutip.Qobj`

            A new instance of :class:`qutip.Qobj` that contains only the states
            corresponding to indices that are **not** in `state_inds`.

        .. note::

            Experimental.
        """
        keep_indices = np.array([s not in states_inds
                                 for s in range(self.shape[0])]).nonzero()[0]

        return self.extract_states(keep_indices, normalize=normalize)

    @property
    def iscp(self):
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type in ["super", "oper"]:
            try:
                eigs = (
                    self
                    # We can test with either Choi or chi, since the basis
                    # transformation between them is unitary and hence
                    # preserves the CP and TP conditions.
                    if self.superrep in ('choi', 'chi')
                    else sr.to_choi(self)
                ).eigenenergies()
                return all(eigs >= 0)
            except:
                return False
        else:
            return False

    @property
    def istp(self):

        if self.type in ["super", "oper"]:
            try:
                # We use the condition from John Watrous' lecture notes,
                # Tr_1(J(Phi)) = identity_2.
                tr_oper = ptrace((
                    self
                    # We can test with either Choi or chi, since the basis
                    # transformation between them is unitary and hence
                    # preserves the CP and TP conditions.
                    if self.superrep in ('choi', 'chi')
                    else sr.to_choi(self)
                ), (0,))
                ident = ops.identity(tr_oper.shape[0])
                return isequal(tr_oper, ident)
            except:
                return False
        else:
            return False

    @property
    def iscptp(self):
        from qutip.superop_reps import to_choi
        if self.type == "super" or self.type == "oper":
            reps = ('choi', 'chi')
            q_oper = to_choi(self) if self.superrep not in reps else self
            return q_oper.iscp and q_oper.istp
        else:
            return False

    @property
    def isherm(self):

        if self._isherm is not None:
            # used previously computed value
            return self._isherm

        if self.dims[0] != self.dims[1]:
            self._isherm = False
        else:
            data = self.data
            h = np.abs((data.transpose().conj() - data).data)
            self._isherm = False if np.any(h > settings.atol) else True

        return self._isherm

    @isherm.setter
    def isherm(self, isherm):
        self._isherm = isherm

    @property
    def type(self):

        if not self._type:
            if self.isoper:
                self._type = 'oper'
            elif self.isket:
                self._type = 'ket'
            elif self.isbra:
                self._type = 'bra'
            elif self.issuper:
                self._type = 'super'
            elif self.isoperket:
                self._type = 'operator-ket'
            elif self.isoperbra:
                self._type = 'operator-bra'
            else:
                self._type = 'other'

        return self._type

    @property
    def shape(self):
        return [np.prod(self.dims[0]), np.prod(self.dims[1])]

    @property
    def isbra(self):
        return (np.prod(self.dims[0]) == 1 and
                isinstance(self.dims[1], list) and
                isinstance(self.dims[1][0], (int, np.integer)))

    @property
    def isket(self):
        return (np.prod(self.dims[1]) == 1 and
                isinstance(self.dims[0], list) and
                isinstance(self.dims[0][0], (int, np.integer)))

    @property
    def isoperbra(self):
        return (np.prod(self.dims[0]) == 1 and
                isinstance(self.dims[1], list) and
                isinstance(self.dims[1][0], list))

    @property
    def isoperket(self):
        return (np.prod(self.dims[1]) == 1 and
                isinstance(self.dims[0], list) and
                isinstance(self.dims[0][0], list))

    @property
    def isoper(self):
        return (isinstance(self.dims[0], list) and
                isinstance(self.dims[0][0], (int, np.integer)) and
                self.dims[0] == self.dims[1])

    @property
    def issuper(self):
        return (isinstance(self.dims[0], list) and
                isinstance(self.dims[0][0], list) and
                self.dims[0] == self.dims[1] and
                self.dims[0][0] == self.dims[1][0])

    @staticmethod
    def evaluate(qobj_list, t, args):
        """Evaluate a time-dependent quantum object in list format. For
        example,

            qobj_list = [H0, [H1, func_t]]

        is evaluated to

            Qobj(t) = H0 + H1 * func_t(t, args)

        and

            qobj_list = [H0, [H1, 'sin(w * t)']]

        is evaluated to

            Qobj(t) = H0 + H1 * sin(args['w'] * t)

        Parameters
        ----------

        qobj_list : list
            A nested list of Qobj instances and corresponding time-dependent
            coefficients.

        t : float
            The time for which to evaluate the time-dependent Qobj instance.

        args : dictionary
            A dictionary with parameter values required to evaluate the
            time-dependent Qobj intance.

        Returns
        -------

        output : Qobj

            A Qobj instance that represents the value of qobj_list at time t.

        """

        q_sum = 0
        if isinstance(qobj_list, Qobj):
            q_sum = qobj_list
        elif isinstance(qobj_list, list):
            for q in qobj_list:
                if isinstance(q, Qobj):
                    q_sum += q
                elif (isinstance(q, list) and len(q) == 2 and
                      isinstance(q[0], Qobj)):
                    if isinstance(q[1], types.FunctionType):
                        q_sum += q[0] * q[1](t, args)
                    elif isinstance(q[1], str):
                        args['t'] = t
                        q_sum += q[0] * float(eval(q[1], globals(), args))
                    else:
                        raise TypeError('Unrecognized format for ' +
                                        'specification of time-dependent Qobj')
                else:
                    raise TypeError('Unrecognized format for specification ' +
                                    'of time-dependent Qobj')
        else:
            raise TypeError(
                'Unrecongized format for specification of time-dependent Qobj')

        return q_sum


# -----------------------------------------------------------------------------
# This functions evaluates a time-dependent quantum object on the list-string
# and list-function formats that are used by the time-dependent solvers.
# Although not used directly in by those solvers, it can for test purposes be
# conventient to be able to evaluate the expressions passed to the solver for
# arbitrary value of time. This function provides this functionality.
#
def qobj_list_evaluate(qobj_list, t, args):
    """
    Depracated: See Qobj.evaluate
    """
    warnings.warn("Deprecated: Use Qobj.evaluate", DeprecationWarning)
    return Qobj.evaluate(qobj_list, t, args)


# -----------------------------------------------------------------------------
#
# A collection of tests used to determine the type of quantum objects, and some
# functions for increased compatibility with quantum optics toolbox.
#

def dag(A):
    """Adjont operator (dagger) of a quantum object.

    Parameters
    ----------
    A : qobj
        Input quantum object.

    Returns
    -------
    oper : qobj
        Adjoint of input operator

    Notes
    -----
    This function is for legacy compatibility only. It is recommended to use
    the ``dag()`` Qobj method.
    """
    if not isinstance(A, Qobj):
        raise TypeError("Input is not a quantum object")

    return A.dag()


def ptrace(Q, sel):
    """Partial trace of the Qobj with selected components remaining.

    Parameters
    ----------
    Q : qobj
        Composite quantum object.
    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    Returns
    -------
    oper: qobj
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


def dims(inpt):
    """Returns the dims attribute of a quantum object.

    Parameters
    ----------
    inpt : qobj
        Input quantum object.

    Returns
    -------
    dims : list
        A ``list`` of the quantum objects dimensions.

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.dims`
    attribute is recommended.
    """
    if isinstance(inpt, Qobj):
        return inpt.dims
    else:
        raise TypeError("Input is not a quantum object")


def shape(inpt):
    """Returns the shape attribute of a quantum object.

    Parameters
    ----------
    inpt : qobj
        Input quantum object.

    Returns
    -------
    shape : list
        A ``list`` of the quantum objects shape.

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.shape`
    attribute is recommended.
    """
    if isinstance(inpt, Qobj):
        return Qobj.shape
    else:
        return np.shape(inpt)


def isket(Q):
    """
    Determines if given quantum object is a ket-vector.

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isket : bool
        True if qobj is ket-vector, False otherwise.

    Examples
    --------
    >>> psi = basis(5,2)
    >>> isket(psi)
    True

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isket`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isket else False


def isbra(Q):
    """Determines if given quantum object is a bra-vector.

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isbra : bool
        True if Qobj is bra-vector, False otherwise.

    Examples
    --------
    >>> psi = basis(5,2)
    >>> isket(psi)
    False

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isbra`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isbra else False


def isoperket(Q):
    """Determines if given quantum object is an operator in column vector form
    (operator-ket).

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isoperket : bool
        True if Qobj is operator-ket, False otherwise.

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isoperket`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isoperket else False


def isoperbra(Q):
    """Determines if given quantum object is an operator in row vector form
    (operator-bra).

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isoperbra : bool
        True if Qobj is operator-bra, False otherwise.

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isoperbra`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isoperbra else False


def isoper(Q):
    """Determines if given quantum object is a operator.

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isoper : bool
        True if Qobj is operator, False otherwise.

    Examples
    --------
    >>> a = destroy(5)
    >>> isoper(a)
    True

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isoper`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isoper else False


def issuper(Q):
    """Determines if given quantum object is a super-operator.

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    issuper  : bool
        True if Qobj is superoperator, False otherwise.

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.issuper`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.issuper else False


def isequal(A, B, tol=None):
    """Determines if two qobj objects are equal to within given tolerance.

    Parameters
    ----------
    A : qobj
        Qobj one
    B : qobj
        Qobj two
    tol : float
        Tolerence for equality to be valid

    Returns
    -------
    isequal : bool
        True if qobjs are equal, False otherwise.

    Notes
    -----
    This function is for legacy compatibility only. Instead, it is recommended
    to use the equality operator of Qobj instances instead: A == B.
    """
    if tol is None:
        tol = settings.atol

    if not isinstance(A, Qobj) or not isinstance(B, Qobj):
        return False

    if A.dims != B.dims:
        return False

    Adat = A.data
    Bdat = B.data
    elems = (Adat - Bdat).data
    if np.any(abs(elems) > tol):
        return False

    return True


def isherm(Q):
    """Determines if given operator is Hermitian.

    Parameters
    ----------
    Q : qobj
        Quantum object

    Returns
    -------
    isherm : bool
        True if operator is Hermitian, False otherwise.

    Examples
    --------
    >>> a = destroy(4)
    >>> isherm(a)
    False

    Notes
    -----
    This function is for legacy compatibility only. Using the `Qobj.isherm`
    attribute is recommended.
    """
    return True if isinstance(Q, Qobj) and Q.isherm else False


# TRAILING IMPORTS
# We do a few imports here to avoid circular dependencies.
from qutip.eseries import eseries
import qutip.superop_reps as sr
import qutip.operators as ops
