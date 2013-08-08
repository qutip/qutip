# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""Main module for QuTiP, consisting of the Quantum Object (Qobj) class and
its methods.
"""
import types
import pickle

# import math functions from numpy.math: required for td string evaluation
from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   ceil, copysign, cos, cosh, degrees, e, exp, expm1, fabs,
                   floor, fmod, frexp, hypot, isinf, isnan, ldexp, log, log10,
                   log1p, modf, pi, radians, sin, sinh, sqrt, tan, tanh, trunc)

from numpy import prod, allclose, shape, where

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import qutip.settings as qset
from qutip._version import version as qversion
from qutip.ptrace import _ptrace
from qutip.permute import _permute
from qutip.sparse import (sp_eigs, _sp_expm, _sp_fro_norm, _sp_max_norm,
                          _sp_one_norm, _sp_L2_norm, _sp_inf_norm)


class Qobj():
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
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', or 'super'.


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
    matrix_element(bra,ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(oper_norm='tr',sparse=False,tol=0,maxiter=100000)
        Returns norm of operator.
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
    transform(inpt,inverse=False)
        Performs a basis transformation defined by `inpt` matrix.
    unit(oper_norm='tr',sparse=False,tol=0,maxiter=100000)
        Returns normalized quantum object.


    """
    __array_priority__ = 100  # sets Qobj priority above numpy arrays

    def __init__(self, inpt=np.array([[0]]), dims=[[], []], shape=[],
                 type=None, isherm=None, fast=False):
        """
        Qobj constructor.
        """
        if fast == 'mc':
            # fast Qobj construction for use in mcsolve with ket output
            self.data = sp.csr_matrix(inpt, dtype=complex)
            self.dims = dims
            self.shape = shape
            self.isherm = False
            self.type = 'ket'
            return

        if fast == 'mc-dm':
            # fast Qobj construction for use in mcsolve with dm output
            self.data = sp.csr_matrix(inpt, dtype=complex)
            self.dims = dims
            self.shape = shape
            self.isherm = True
            self.type = 'oper'
            return

        elif isinstance(inpt, Qobj):
            # if input is already Qobj then return identical copy

            # make sure matrix is sparse (safety check)
            self.data = sp.csr_matrix(inpt.data, dtype=complex)

            if not np.any(dims):
                # Dimensions of quantum object used for keeping track of tensor
                # components
                self.dims = inpt.dims
            else:
                self.dims = dims

            if not np.any(shape):
                # Shape of undelying quantum obejct data matrix
                self.shape = inpt.shape
            else:
                self.shape = shape

        else:
            # if input is int, float, or complex then convert to array
            if isinstance(inpt, (int, float, complex, np.int64)):
                inpt = np.array([[inpt]])

            # case where input is array or sparse
            if (isinstance(inpt, np.ndarray)) or sp.issparse(inpt):

                if inpt.ndim == 1:
                    inpt = inpt[:, np.newaxis]

                self.data = sp.csr_matrix(inpt, dtype=complex)
 
                if not np.any(dims):
                    self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
                else:
                    self.dims = dims

                if not np.any(shape):
                    self.shape = [int(inpt.shape[0]), int(inpt.shape[1])]
                else:
                    self.shape = shape

            elif isinstance(inpt, list):
                # case where input is not array or sparse, i.e. a list
                if len(np.array(inpt).shape) == 1:
                    # if list has only one dimension (i.e [5,4])
                    inpt = np.array([inpt])
                else:  # if list has two dimensions (i.e [[5,4]])
                    inpt = np.array(inpt)

                self.data = sp.csr_matrix(inpt, dtype=complex)

                if not np.any(dims):
                    self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
                else:
                    self.dims = dims

                if not np.any(shape):
                    self.shape = [int(inpt.shape[0]), int(inpt.shape[1])]
                else:
                    self.shape = shape

            else:
                print("Warning: Initializing Qobj from unsupported type")
                inpt = np.array([[0]])
                self.data = sp.csr_matrix(inpt, dtype=complex)
                self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
                self.shape = [int(inpt.shape[0]), int(inpt.shape[1])]

        # Signifies if quantum object corresponds to Hermitian operator
        if isherm is None:
            if qset.auto_herm:
                self.isherm = hermcheck(self)
            else:
                self.isherm = None
        else:
            self.isherm = isherm
        # Signifies if quantum object corresponds to a ket, bra, operator, or
        # super-operator
        if type is None:
            self.type = ischeck(self)
        else:
            self.type = type

    def __add__(self, other):  # defines left addition for Qobj class
        """
        ADDITION with Qobj on LEFT [ ex. Qobj+4 ]
        """
        if _checkeseries(other) == 'eseries':
            return other.__radd__(self)
        if not isinstance(other, Qobj):
            other = Qobj(other)
        if np.prod(other.shape) == 1 and np.prod(self.shape) != 1:
            # case for scalar quantum object
            dat = np.array(other.full())[0][0]
            if dat != 0:
                out = Qobj(type=self.type)
                if self.type in ['oper', 'super']:
                    out.data = self.data + dat * sp.identity(
                        self.shape[0], dtype=complex, format='csr')
                else:
                    out.data = self.data
                    out.data.data = out.data.data + dat
                out.dims = self.dims
                out.shape = self.shape
                isherm = None
                if isinstance(dat, (int, float)):
                    isherm = self.isherm
                if qset.auto_tidyup:
                    return Qobj(out, type=self.type, isherm=isherm).tidyup()
                else:
                    return Qobj(out, type=self.type, isherm=isherm)
            else:  # if other qobj is zero object
                return self
        elif np.prod(self.shape) == 1 and np.prod(other.shape) != 1:
            # case for scalar quantum object
            dat = np.array(self.full())[0][0]
            if dat != 0:
                out = Qobj()
                if other.type in ['oper', 'super']:
                    out.data = dat * sp.identity(other.shape[0], dtype=complex,
                                                 format='csr') + other.data
                else:
                    out.data = other.data
                    out.data.data = out.data.data + dat
                out.dims = other.dims
                out.shape = other.shape
                isherm = None
                if isinstance(dat, (int, float)):
                    isherm = other.isherm
                if qset.auto_tidyup:
                    return Qobj(out, type=other.type, isherm=isherm).tidyup()
                else:
                    return Qobj(out, type=other.type, isherm=isherm)
            else:
                return other
        elif self.dims != other.dims:
            raise TypeError('Incompatible quantum object dimensions')
        elif self.shape != other.shape:
            raise TypeError('Matrix shapes do not match')
        else:  # case for matching quantum objects
            out = Qobj(type=self.type)
            out.data = self.data + other.data
            out.dims = self.dims
            out.shape = self.shape
            isherm = None
            if self.type in np.array(['ket', 'bra', 'super']):
                isherm = False
            elif self.isherm and self.isherm == other.isherm:
                isherm = True
            elif ((self.isherm and not other.isherm) or
                  (not self.isherm and other.isherm)):
                isherm = False
            if qset.auto_tidyup:
                return Qobj(out, type=self.type, isherm=isherm).tidyup()
            else:
                return Qobj(out, type=self.type, isherm=isherm)

    def __radd__(self, other):
        """
        ADDITION with Qobj on RIGHT [ ex. 4+Qobj ] (just calls left addition)
        """
        out = self + other
        if qset.auto_tidyup:
            return out.tidyup()
        else:
            return out

    def __sub__(self, other):
        """
        SUBTRACTION with Qobj on LEFT [ ex. Qobj-4 ]
        """
        out = self + (-other)
        if qset.auto_tidyup:
            return out.tidyup()
        else:
            return out

    def __rsub__(self, other):
        """
        SUBTRACTION with Qobj on RIGHT [ ex. 4-Qobj ]
        """
        out = (-self) + other
        if qset.auto_tidyup:
            return out.tidyup()
        else:
            return out

    def __mul__(self, other):
        """
        MULTIPLICATION with Qobj on LEFT [ ex. Qobj*4 ]
        """
        if isinstance(other, Qobj):  # if both are quantum objects
            if (self.shape[1] == other.shape[0] and
                    self.dims[1] == other.dims[0]):
                out = Qobj()
                out.data = self.data * other.data
                dims = [self.dims[0], other.dims[1]]
                if not isinstance(dims[0][0], list):
                    r = range(len(dims[0]))
                    mask = [dims[0][n] == dims[1][n] == 1 for n in r]
                    out.dims = [max(
                        [1], [dims[0][n] for n in r if not mask[n]]),
                        max([1], [dims[1][n] for n in r if not mask[n]])]
                else:
                    out.dims = dims
                out.shape = [self.shape[0], other.shape[1]]
                out.type = ischeck(out)
                out.isherm = hermcheck(out)
                if qset.auto_tidyup:
                    return out.tidyup()
                else:
                    return out
            else:
                raise TypeError("Incompatible Qobj shapes")

        elif isinstance(other, (list, np.ndarray)):
            # if other is a list, do element-wise multiplication
            return np.array([self * item for item in other])

        elif _checkeseries(other) == 'eseries':
            return other.__rmul__(self)

        elif isinstance(other, (int, float, complex, np.int64)):
            out = Qobj(type=self.type)
            out.data = self.data * other
            out.dims = self.dims
            out.shape = self.shape
            isherm = None
            if isinstance(other, (int, float)):
                isherm = self.isherm
            if qset.auto_tidyup:
                return Qobj(out, type=self.type, isherm=isherm).tidyup()
            else:
                return Qobj(out, type=self.type, isherm=isherm)
        else:
            raise TypeError("Incompatible object for multiplication")

    def __rmul__(self, other):
        """
        MULTIPLICATION with Qobj on RIGHT [ ex. 4*Qobj ]
        """
        if isinstance(other, Qobj):  # if both are quantum objects
            if (self.shape[1] == other.shape[0] and
                    self.dims[1] == other.dims[0]):
                out = Qobj()
                out.data = other.data * self.data
                out.dims = self.dims
                out.shape = [self.shape[0], other.shape[1]]
                out.type = ischeck(out)
                out.isherm = hermcheck(out)
                if qset.auto_tidyup:
                    return out.tidyup()
                else:
                    return out
            else:
                raise TypeError("Incompatible Qobj shapes")

        if isinstance(other, (list, np.ndarray)):
            # if other is a list, do element-wise multiplication
            return np.array([item * self for item in other])

        if _checkeseries(other) == 'eseries':
            return other.__mul__(self)

        if isinstance(other, (int, float, complex, np.int64)):
            out = Qobj(type=self.type)
            out.data = other * self.data
            out.dims = self.dims
            out.shape = self.shape
            isherm = None
            if isinstance(other, (int, float)):
                isherm = self.isherm
            if qset.auto_tidyup:
                return Qobj(out, type=self.type, isherm=isherm).tidyup()
            else:
                return Qobj(out, type=self.type, isherm=isherm)
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
        # if isinstance(other,Qobj): #if both are quantum objects
        # raise TypeError("Incompatible Qobj shapes [division with Qobj not
        # implemented]")
        if isinstance(other, (int, float, complex, np.int64)):
            out = Qobj(type=self.type)
            out.data = self.data / other
            out.dims = self.dims
            out.shape = self.shape
            isherm = None
            if isinstance(other, (int, float, np.int64)):
                isherm = self.isherm
            if qset.auto_tidyup:
                return Qobj(out, type=self.type, isherm=isherm).tidyup()
            else:
                return Qobj(out, type=self.type, isherm=isherm)
        else:
            raise TypeError("Incompatible object for division")

    def __neg__(self):
        """
        NEGATION operation.
        """
        out = Qobj()
        out.data = -self.data
        out.dims = self.dims
        out.shape = self.shape
        if qset.auto_tidyup:
            return Qobj(out, type=self.type, isherm=self.isherm).tidyup()
        else:
            return Qobj(out, type=self.type, isherm=self.isherm)

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
                self.shape == other.shape and
                abs(_sp_max_norm(self - other)) < 1e-14):
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
            out = Qobj(data, dims=self.dims, shape=self.shape)
            if qset.auto_tidyup:
                return out.tidyup()
            else:
                return out
        except:
            raise ValueError('Invalid choice of exponent.')

    def __str__(self):
        s = ""
        if self.type == 'oper' or self.type == 'super':
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type +
                  ", isherm = " + str(self.isherm) + "\n")
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type + "\n")
        s += "Qobj data =\n"
        if all(np.imag(self.data.data) == 0):
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
        self.__dict__.update({'qutip_version': qversion[:5]})
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
        s = r'\begin{equation}\text{'
        if self.type == 'oper' or self.type == 'super':
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type +
                  ", isHerm = " + str(self.isherm))
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type)

        s += r'}\\[1em]'

        M, N = self.data.shape

        s += r'\begin{pmatrix}'

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
                if abs(np.imag(d)) < 1e-12:
                    return s + _format_float(np.real(d))
                elif abs(np.real(d)) < 1e-12:
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

        s += r'\end{pmatrix}\end{equation}'
        return s

    def dag(self):
        """Returns the adjoint operator of quantum object.
        """
        out = Qobj()
        out.data = self.data.T.conj()
        out.dims = [self.dims[1], self.dims[0]]
        out.shape = [self.shape[1], self.shape[0]]
        out.isherm = self.isherm
        out.type = ischeck(out)
        return out

    def conj(self):
        """Returns the conjugate operator of quantum object.

        """
        out = Qobj(type=self.type)
        out.data = self.data.conj()
        out.dims = [self.dims[1], self.dims[0]]
        out.shape = [self.shape[1], self.shape[0]]
        return out

    def norm(self, oper_norm='tr', sparse=False, tol=0, maxiter=100000):
        """Returns the norm of a quantum object.

        Norm is L2-norm for kets and trace-norm (by default) for operators.
        Other operator norms may be specified using the `oper_norm` argument.

        Parameters
        ----------
        oper_norm : str
            Which norm to use for operators: trace 'tr', Frobius 'fro',
            one 'one', or max 'max'. This parameter does not affect the norm
            of a state vector.

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
        if self.type == 'oper' or self.type == 'super':
            if oper_norm == 'tr':
                vals = sp_eigs(
                    self, vecs=False, sparse=sparse, tol=tol, maxiter=maxiter)
                return sum(sqrt(abs(vals) ** 2))
            elif oper_norm == 'fro':
                return _sp_fro_norm(self)
            elif oper_norm == 'one':
                return _sp_one_norm(self)
            elif oper_norm == 'max':
                return _sp_max_norm(self)
            else:
                raise ValueError(
                    "Operator norm must be 'tr', 'fro', 'one', or 'max'.")
        else:
            return _sp_L2_norm(self)

    def tr(self):
        """The trace of a quantum object.

        Returns
        -------
        trace: float
            Returns ``real`` if operator is Hermitian, returns ``complex``
            otherwise.

        """
        if self.isherm:
            return float(np.real(sum(self.data.diagonal())))
        else:
            return complex(sum(self.data.diagonal()))

    def full(self):
        """Returns a dense array from quantum object.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.

        """
        return np.array(self.data.todense())

    def diag(self):
        """Diagonal elements of Qobj.

        Returns
        -------
        diags: array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.

        """
        out = self.data.diagonal()
        if np.any(np.imag(out) > 1e-15) or not self.isherm:
            return out
        else:
            return np.real(out)

    def expm(self):
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Returns
        -------
        oper : qobj
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.

        """
        if self.dims[0][0] == self.dims[1][0]:
            F = _sp_expm(self)
            out = Qobj(F, dims=self.dims, shape=self.shape)
            return out.tidyup() if qset.auto_tidyup else out
        else:
            raise TypeError('Invalid operand for matrix exponential')

    def checkherm(self):
        """ Explicitly check if the Qobj is hermitian

        Returns
        -------
        isherm: bool
            Returns the new value of isherm property.
        """
        self.isherm = hermcheck(self)
        return self.isherm

    def sqrtm(self, sparse=False, tol=0, maxiter=100000):
        """The sqrt of a quantum operator.

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
            evals, evecs = sp_eigs(
                self, sparse=sparse, tol=tol, maxiter=maxiter)
            numevals = len(evals)
            dV = sp.spdiags(
                np.sqrt(np.abs(evals)), 0, numevals, numevals, format='csr')
            evecs = sp.hstack(evecs, format='csr')
            spDv = dV.dot(evecs.conj().T)
            out = Qobj(evecs.dot(spDv), dims=self.dims, shape=self.shape)
            if qset.auto_tidyup:
                return out.tidyup()
            else:
                return out
        else:
            raise TypeError('Invalid operand for matrix square root')

    def unit(self, oper_norm='tr', sparse=False, tol=0, maxiter=100000):
        """Returns the operator or state normalized to unity.

        Uses norm from Qobj.norm().

        Parameters
        ----------
        oper_norm : str
            Requested norm for operator. Norm is always L2 for states.
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
        out = self / self.norm(
            oper_norm=oper_norm, sparse=sparse, tol=tol, maxiter=maxiter)
        if qset.auto_tidyup:
            return out.tidyup()
        else:
            return out

    def ptrace(self, sel):
        """Partial trace of the Qobj with selected components remaining.

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
        qdata, qdims, qshape = _ptrace(self, sel)
        if qset.auto_tidyup:
            return Qobj(qdata, qdims, qshape).tidyup()
        else:
            return Qobj(qdata, qdims, qshape)

    def permute(self, order):
        """
        Permutes a composite quantum object in the given order.

        Parameters
        ----------
        order : list/array
            List specifying new tensor order.

        Returns
        -------
        P : qobj
            Permuted quantum object.

        """
        data, dims, shape = _permute(self, order)
        return Qobj(data, dims=dims, shape=shape)

    def tidyup(self, atol=qset.auto_tidyup_atol):
        """Removes small elements from a quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup.  Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper: qobj
            Quantum object with small elements removed.

        """
        out = Qobj(dims=self.dims, shape=self.shape,
                   type=self.type, isherm=self.isherm)

        abs_data = abs(self.data.data.flatten())
        if np.any(abs_data):
            mx = max(abs_data)
            if mx >= atol:
                data = abs(self.data.data)
                out.data = self.data.copy()
                out.data.data[data < (atol * mx + np.finfo(float).eps)] = 0
            else:
                out.data = sp.csr_matrix(
                    (self.shape[0], self.shape[1]), dtype=complex)
        else:
            out.data = sp.csr_matrix(
                (self.shape[0], self.shape[1]), dtype=complex)

        out.data.eliminate_zeros()
        return out

    def transform(self, inpt, inverse=False):
        """Performs a basis transformation defined by an input array.

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
        if isinstance(inpt, list) or isinstance(inpt, np.ndarray):
            if len(inpt) != max(self.shape):
                raise TypeError(
                    'Invalid size of ket list for basis transformation')
            S = np.matrix(np.hstack([psi.full() for psi in inpt]))
        elif isinstance(inpt, np.ndarray):
            S = np.matrix(inpt)
        else:
            raise TypeError('Invalid operand for basis transformation')

        # normalize S just in case the supplied basis states aren't normalized
        # S = S/la.norm(S)

        out = Qobj(type=self.type, dims=self.dims, shape=self.shape)
        out.isherm = self.isherm
        out.type = self.type

        # transform data
        if inverse:
            if isket(self):
                out.data = S.H * self.data
            elif isbra(self):
                out.data = self.data * S
            else:
                out.data = S.H * self.data * S
        else:
            if isket(self):
                out.data = S * self.data
            elif isbra(self):
                out.data = self.data * S.H
            else:
                out.data = S * self.data * S.H

        # force sparse
        out.data = sp.csr_matrix(out.data, dtype=complex)

        return out

    def matrix_element(self, bra, ket):
        """Calculates a matrix element.

        Gives matrix for the Qobj sandwiched between a `bra` and `ket`
        vector.

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

        if isoper(self):
            if isbra(bra) and isket(ket):
                return (bra.data * self.data * ket.data)[0, 0]

            if isket(bra) and isket(ket):
                return (bra.data.T * self.data * ket.data)[0, 0]

        raise TypeError("Can only calculate matrix elements for operators " +
                        "and between ket and bra Qobj")

    def overlap(self, state):
        """Calculates a overlap between two state vectors.

        Gives overlap (scalar product) for the Qobj and `state` state vector.

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

        if isbra(self):
            if isket(state):
                return (self.data * state.data)[0, 0]
            elif isbra(state):
                return (self.data * state.data.H)[0, 0]

        elif isket(self):
            if isbra(state):
                return (self.data.H * state.data.H)[0, 0]
            elif isket(state):
                return (self.data.H * state.data)[0, 0]

        raise TypeError("Can only calculate overlap for state vector Qobjs")

    def eigenstates(self, sparse=False, sort='low',
                    eigvals=0, tol=0, maxiter=100000):
        """Find the eigenstates and eigenenergies.

        Eigenstates and Eigenvalues are defined for operators and
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
        evals, evecs = sp_eigs(self, sparse=sparse, sort=sort,
                               eigvals=eigvals, tol=tol, maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        new_shape = [self.shape[0], 1]
        ekets = np.array(
            [Qobj(vec, dims=new_dims, shape=new_shape) for vec in evecs])
        norms = np.array([ket.norm() for ket in ekets])
        return evals, ekets / norms

    def eigenenergies(self, sparse=False, sort='low',
                      eigvals=0, tol=0, maxiter=100000):
        """Finds the Eigenenergies (Eigenvalues) of a quantum object.

        Eigenenergies are defined for operators or superoperators only.

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
        return sp_eigs(self, vecs=False, sparse=sparse, sort=sort,
                       eigvals=eigvals, tol=tol, maxiter=maxiter)

    def groundstate(self, sparse=False, tol=0, maxiter=100000):
        """Finds the ground state Eigenvalue and Eigenvector.

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
        grndval, grndvec = sp_eigs(
            self, sparse=sparse, eigvals=1, tol=tol, maxiter=maxiter)
        new_dims = [self.dims[0], [1] * len(self.dims[0])]
        new_shape = [self.shape[0], 1]
        grndvec = Qobj(grndvec[0], dims=new_dims, shape=new_shape)
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
        out.data = self.data.T
        out.dims = [self.dims[1], self.dims[0]]
        out.shape = [self.shape[1], self.shape[0]]
        out.type = ischeck(out)
        return out

    def extract_states(self, states_indices, normalize=False):
        """
        Create a new Qobj instance for which only states corresponding to
        those in state_indices has been kept.

        Parameters
        ----------
        states_indices : list of integer
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
            corresponding to the indices in `state_indices`.

        .. note::

            Experimental.

        """
        if isoper(self):
            q = Qobj(self.data[states_indices, :][:, states_indices])
        elif isket(self):
            q = Qobj(self.data[states_indices, :])
        elif isbra(self):
            q = Qobj(self.data[:, states_indices])
        else:
            raise TypeError("Can only eliminate states from operators or " +
                            "state vectors")

        return q.unit() if normalize else q

    def eliminate_states(self, states_indices, normalize=False):
        """
        Create a new Qobj instance for which states corresponding to
        those in state_indices has been eliminated.

        Parameters
        ----------
        states_indices : list of integer
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
            corresponding to indices that are **not** in `state_indices`.

        .. note::

            Experimental.
        """
        keep_indices = np.array([s not in states_indices
                                 for s in range(self.shape[0])]).nonzero()[0]

        return self.extract_states(keep_indices, normalize=normalize)

#------------------------------------------------------------------------------
# This functions evaluates a time-dependent quantum object on the list-string
# and list-function formats that are used by the time-dependent solvers.
# Although not used directly in by those solvers, it can for test purposes be
# conventient to be able to evaluate the expressions passed to the solver for
# arbitrary value of time. This function provides this functionality.
#


def qobj_list_evaluate(qobj_list, t, args):
    """
    Evaluate a time-dependent qobj in list format. For example,

        qobj_list = [H0, [H1, func_t]]

    is evaluated to

        Qobj(t) = H0 + H1 * func_t(t, args)

    and

        qobj_list = [H0, [H1, sin(w * t)]]

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

        The Qobj that represents the value of qobj_list at time t.

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
                    raise TypeError('Unrecognized format for specification ' +
                                    'of time-dependent Qobj')
            else:
                raise TypeError('Unrecognized format for specification ' +
                                'of time-dependent Qobj')
    else:
        raise TypeError(
            'Unrecongized format for specification of time-dependent Qobj')

    return q_sum


#------------------------------------------------------------------------------
#
# some functions for increased compatibility with quantum optics toolbox:
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
    This function is for compatibility with the qotoolbox only.
    It is recommended to use the ``dag()`` Qobj method.

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
    Depreciated in QuTiP v. 2.0.

    """
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
    This function is for compatibility with the qotoolbox only.
    Using the `Qobj.dims` attribute is recommended.


    """
    if isinstance(inpt, Qobj):
        return inpt.dims
    else:
        raise TypeError("Incompatible object for dims (not a Qobj)")


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
    This function is for compatibility with the qotoolbox only.
    Using the `Qobj.dims` attribute is recommended.

    """
    if isinstance(inpt, Qobj):
        return Qobj.shape
    else:
        return np.shape(inpt)


#------------------------------------------------------------------------------
#
# functions for storing and loading Qobj instances to files
#

def qobj_save(qobj, filename):
    """
    Saves the given qobj to file 'filename'
    Argument qobj input operator
    Argument filename string for output file name

    Returns file returns qobj as file in current directory
    """
    with open(filename, 'wb') as f:
        pickle.dump(qobj, f, protocol=2)


def qobj_load(filename):
    """
    Loads a quantum object saved using qobj_save
    Argument filename filename of request qobject

    Returns Qobj returns quantum object
    """
    with open(filename, 'wb') as f:
        qobj = pickle.load(f)

    return qobj


#------------------------------------------------------------------------------
#
# check for class type ESERIES
#

def _checkeseries(inpt):
    """
    Checks for ESERIES class types
    """
    from qutip.eseries import eseries
    if isinstance(inpt, eseries):
        return 'eseries'
    else:
        pass


#------------------------------------------------------------------------------
#
# A collection of tests used to determine the type of quantum objects.
#

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
    >>> psi=basis(5,2)
    >>> isket(psi)
    True

    """

    if not isinstance(Q, Qobj):
            return False

    return isinstance(Q.dims[0], list) and prod(Q.dims[1]) == 1


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
    >>> psi=basis(5,2)
    >>> isket(psi)
    False

    """

    if not isinstance(Q, Qobj):
        return False

    return isinstance(Q.dims[1], list) and (prod(Q.dims[0]) == 1)


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
    >>> a=destroy(5)
    >>> isoper(a)
    True

    """

    if not isinstance(Q, Qobj):
        return False

    return (isinstance(Q.dims[0], list) and
            isinstance(Q.dims[0][0], int) and (Q.dims[0] == Q.dims[1]))


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

    """

    if not isinstance(Q, Qobj):
        return False

    result = isinstance(Q.dims[0], list) and isinstance(Q.dims[0][0], list)
    if result:
        result = (Q.dims[0] == Q.dims[1]) and (Q.dims[0][0] == Q.dims[1][0])
    return result


def isequal(A, B, tol=1e-12):
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

    """

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


def ischeck(Q):
    if isoper(Q):
        return 'oper'
    elif isket(Q):
        return 'ket'
    elif isbra(Q):
        return 'bra'
    elif issuper(Q):
        return 'super'
    else:
        return 'other'


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
    >>> a=destroy(4)
    >>> isherm(a)
    False

    """

    if not isinstance(Q, Qobj):
        return False

    if Q.dims[0] != Q.dims[1]:
        return False
    else:
        dat = Q.data
        elems = (dat.transpose().conj() - dat).data
        if np.any(abs(elems) > 1e-12):
            return False
        else:
            return True

hermcheck = isherm
