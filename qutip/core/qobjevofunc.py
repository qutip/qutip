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
"""Time-dependent Quantum Object (QobjEvo) wrapper class
for function returning Qobj.
"""
__all__ = ['QobjEvoFunc']

from .qobj import Qobj
from .qobjevo import QobjEvo, QobjEvoBase
import numpy as np
from .cy.cqobjevo import CQobjFunc


class QobjEvoFunc(QobjEvoBase):
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states from a function or callable.

    The QobjEvoFunc class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : QobjEvo[Func], Qobj
        * : QobjEvo[Func], Qobj, Complex-number
        / : Complex-number
    and some common linear operator/state operations. The QobjEvoFunc is
    constructed from a function that return a Qobj from time and extra
    arguments.

    The signature of the function must be
    - f(t, args) -> Qobj

    args is a dict of (key: object). The keys must be a valid variables string.

    Parameters
    ----------
    QobjEvoFunc(Q_object=[], args={}, copy=True)

    Q_object : callable
        Function/method that return the Qobj at time t.

    args : dictionary that contain the arguments for function.

    copy : bool
        If Q_object is already a QobjEvoFunc, return a copy.

    Attributes
    ----------
    func : callable
        Funtion wrapped to a Qobj.

    cte : Qobj
        Value at time 0. (To obtain the dimensions, etc.)

    args : map
        arguments of the coefficients

    compiled_qobjevo : cy_qobj
        Cython version of the QobjEvo

    const : bool
        Indicates if quantum object is Constant

    operation_stack : list of _Block_transform
        List of operation that are done on the Qobj.
        ex. func -> liouvillian -> + Qobj

    Methods
    -------
    copy() :
        Create copy of QobjEvoFunc

    arguments(new_args):
        Update the args and set the dynamics_args

    Math:
        +/- QobjEvo, Qobj, scalar:
            Addition is possible between QobjEvo and with Qobj or scalar
        -
            Negation operator
        * Qobj, scalar:
            Product is possible with Qobj or scalar
        / scalar:
            It is possible to divide by scalar only

    conj()
        Return the conjugate of quantum object.

    dag()
        Return the adjoint (dagger) of quantum object.

    trans()
        Return the transpose of quantum object.

    norm()
        Return self.dag() * self.
        Only possible if num_obj == 1

    permute(order)
        Returns composite qobj with indices reordered.

    &
        tensor between Quantum Object

    linear_map(f, op_mapping)
        Apply a transformation function to the Quantum Object

    __call__(t, args={}, data=False):
        Return the Qobj at time t.

    mul(t, mat):
        Product of this at t time with a Qobj or np.ndarray.

    expect(t, psi, herm=False):
        Calculates the expectation value for the quantum object (if operator,
            no check) and state psi.
        Return only the real part if `herm` is True.
    """
    def __init__(self, Q_object=[], args={}, copy=True):
        if isinstance(Q_object, QobjEvoFunc):
            if copy:
                self.cte = Q_object.cte.copy()
                self.args = Q_object.args.copy()
                self.operation_stack = [oper.copy()
                                        for oper in Q_object.operation_stack]
                self._shifted = Q_object._shifted
                self.func = Q_object.func
                self.const = False
                self.compiled_qobjevo = CQobjFunc(self)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args, state, e_ops)
            return

        if not callable(Q_object):
            raise TypeError("expected a function")
        self.func = Q_object
        self.cte = self.func(0, args)
        if not isinstance(self.cte, Qobj):
            raise TypeError("QobjEvoFunc require a "
                            "function returning a Qobj")

        self.args = args.copy() if copy else args

        self.compiled_qobjevo = CQobjFunc(self)
        self.operation_stack = []
        self._shifted = False
        self.const = False

    def __call__(self, t, args={}, data=False):
        try:
            t = float(t)
        except Exception as e:
            raise TypeError("t should be a real scalar.") from e
        out = self._get_qobj(t, args)
        if data:
            out = out.data
        return out

    def _get_qobj(self, t, args={}):
        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        if self._shifted:
            t += now_args["_t0"]
        qobj = self.func(t, now_args)
        for transform in self.operation_stack:
            qobj = transform(qobj, t, now_args)
        return qobj

    def copy(self):
        new = QobjEvoFunc.__new__(QobjEvoFunc)
        new.__dict__ = self.__dict__.copy()
        new.compiled_qobjevo = CQobjFunc(new)
        new.args = self.args.copy()
        new.cte = self.cte.copy()
        new.operation_stack = [oper.copy() for oper in self.operation_stack]
        return new

    def arguments(self, new_args, state=None, e_ops=[]):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(new_args)

    # Math function
    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, QobjEvoBase):
            self.operation_stack.append(_Block_Sum_Qoe(other))
        elif isinstance(other, Qobj):
            self.operation_stack.append(_Block_Sum(other))
        else:
            self.operation_stack.append(_Block_Sum(other))
        if not self._check_validity():
            self.operation_stack = self.operation_stack[:-1]
            return NotImplemented
        return self

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = -self.copy()
        res += other
        return res

    def __isub__(self, other):
        self += (-other)
        return self

    def __mul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):
        res = self.copy()
        if isinstance(other, QobjEvoBase):
            res.operation_stack.append(_Block_rmul_Qoe(other))
        else:
            res.operation_stack.append(_Block_rmul(other))
        if not self._check_validity():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
        return res

    def __imul__(self, other):
        if isinstance(other, QobjEvoBase):
            self.operation_stack.append(_Block_mul_Qoe(other))
        else:
            self.operation_stack.append(_Block_mul(other))
        if not self._check_validity():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
        return self

    def __matmul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmatmul__(self, other):
        res = self.copy()
        if isinstance(other, QobjEvoBase):
            res.operation_stack.append(_Block_rmul_Qoe(other))
        else:
            res.operation_stack.append(_Block_rmul(other))
        if not self._check_validity():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
        return res


    def __imatmul__(self, other):
        if isinstance(other, QobjEvoBase):
            self.operation_stack.append(_Block_mul_Qoe(other))
        else:
            self.operation_stack.append(_Block_mul(other))
        if not self._check_validity():
            self.operation_stack = self.operation_stack[:-1]
            raise Exception
        return self


    def __div__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            res = self.copy()
            res *= other**(-1)
            return res
        else:
            raise TypeError('Incompatible object for division')

    def __idiv__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            self *= other**(-1)
        else:
            raise TypeError('Incompatible object for division')
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.operation_stack.append(_Block_neg())
        res._check_validity()
        return res

    def __and__(self, other):
        """
        Syntax shortcut for tensor:
        A & B ==> tensor(A, B)
        """
        return self._tensor(other)

    # Transformations
    def trans(self):
        res = self.copy()
        res.operation_stack.append(_Block_trans())
        res._check_validity()
        return res

    def conj(self):
        res = self.copy()
        res.operation_stack.append(_Block_conj())
        res._check_validity()
        return res

    def dag(self):
        res = self.copy()
        res.operation_stack.append(_Block_dag())
        res._check_validity()
        return res

    def _cdc(self):
        """return a.dag * a """
        res = self.copy()
        res.operation_stack.append(_Block_cdc())
        res._check_validity()
        return res

    def _tensor(self, other):
        res = self.copy()
        res.operation_stack.append(_Block_tensor_l(other))
        res._check_validity()
        return res

    def _tensor_left(self, other):
        res = self.copy()
        res.operation_stack.append(_Block_tensor_r(other))
        res._check_validity()
        return res

    def _spre(self):
        res = self.copy()
        res.operation_stack.append(_Block_pre())
        res._check_validity()
        return res

    def _spost(self):
        res = self.copy()
        res.operation_stack.append(_Block_post())
        res._check_validity()
        return res

    def _lindblad_dissipator(self, chi=0):
        res = self.copy()
        chi = 0 if chi is None else chi
        res.operation_stack.append(_Block_lindblad_dissipator(chi))
        res._check_validity()
        return res

    def _prespostdag(self):
        """return spre(a) * spost(a.dag()) """
        res = self.copy()
        res.operation_stack.append(_Block_prespostdag())
        res._check_validity()
        return res

    def _liouvillian_h(self):
        """return 1j * (spre(a) - spost(a)) """
        res = self.copy()
        res.operation_stack.append(_Block_liouvillian_H())
        res._check_validity()
        return res

    def _shift(self):
        """shift t by args("_t0") """
        res = self.copy()
        res._shifted = True
        res.args["_t0"] = 0
        res._check_validity()
        return res

    # Unitary function of Qobj
    def tidyup(self, atol=1e-12):
        for transform in self.operation_stack:
            transform.tidyup(atol)
        return self

    def compress(self):
        pass

    def permute(self, order):
        """
        Permute tensor subspaces of the quantum object
        """
        res = self.copy()
        res.operation_stack.append(_Block_permute(order))
        return res

    def _check_validity(self):
        try:
            self.cte = self.__call__(0)
            self.compiled_qobjevo.reset_shape()
        except Exception:
            return False
        return True

    # function to apply custom transformations
    def linear_map(self, op_mapping):
        """
        Apply mapping to each Qobj contribution.
        """
        if op_mapping is spre:
            return self._spre()
        if op_mapping is spost:
            return self._spost()

        res = self.copy()
        res.operation_stack.append(_Block_linear_map(op_mapping))
        res._check_validity()
        return res

    def to(self, data_type):
        """
        Convert the underlying data store of all component into the desired
        storage representation.

        The different storage representations available are the "data-layer
        types".  By default, these are `qutip.data.Dense` and `qutip.data.CSR`,
        which respectively construct a dense matrix store and a compressed
        sparse row one.

        The `QobjEvo` is transformed inplace.

        Arguments
        ---------
        data_type : type
            The data-layer type that the data of this `Qobj` should be
            converted to.

        Returns
        -------
        None
        """
        res = self.copy()
        res.operation_stack.append(_Block_to(data_type))
        res._check_validity()
        return res


class _Block_transform:
    def __init__(self, other=None):
        self.other = other

    def tidyup(self, atol):
        pass

    def copy(self):
        return self.__class__(self.other)

    def __call__(self, obj, t, args={}):
        return obj


class _Block_neg(_Block_transform):
    def __call__(self, obj, t, args={}):
        return -obj


class _Block_Sum(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj + self.other


class _Block_mul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj * self.other


class _Block_rmul(_Block_transform):
    def __call__(self, obj, t, args={}):
        return self.other * obj


class _Block_Sum_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj + self.other(t, args=args)


class _Block_mul_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj * self.other(t, args=args)


class _Block_rmul_Qoe(_Block_transform):
    def __call__(self, obj, t, args={}):
        return self.other(t, args=args) * obj


class _Block_tensor_l(_Block_transform):
    def __call__(self, obj, t, args={}):
        from .tensor import tensor
        if isinstance(self.other, QobjEvoBase):
            return tensor(obj, self.other(t, args))
        return tensor(obj, self.other)


class _Block_tensor_r(_Block_transform):
    def __call__(self, obj, t, args={}):
        from .tensor import tensor
        if isinstance(self.other, QobjEvoBase):
            return tensor(self.other(t, args), obj)
        return tensor(self.other, obj)


class _Block_trans(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.trans()


class _Block_conj(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.conj()


class _Block_dag(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.dag()


class _Block_cdc(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.dag() * obj


class _Block_prespostdag(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spre(obj) * spost(obj.dag())


class _Block_permute(_Block_transform):
    def __call__(self, obj, t, args={}):
        return obj.permute(self.other)


class _Block_pre(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spre(obj)


class _Block_post(_Block_transform):
    def __call__(self, obj, t, args={}):
        return spost(obj)


class _Block_liouvillian_H(_Block_transform):
    def __call__(self, obj, t, args={}):
        return -1.0j * (spre(obj) - spost(obj))


class _Block_lindblad_dissipator(_Block_transform):
    def __call__(self, obj, t, args={}):
        return lindblad_dissipator(obj, chi=self.other)


class _Block_linear_map(_Block_transform):
    def __init__(self, func):
        self.func = func

    def copy(self):
        return _Block_linear_map(self.func)

    def __call__(self, obj, t, args={}):
        return self.func(obj)


class _Block_to(_Block_transform):
    def __init__(self, data_type):
        self.data_type = data_type

    def copy(self):
        return _Block_to(self.data_type)

    def __call__(self, obj, t, args={}):
        return obj.to(self.data_type)

from .superoperator import spre, spost, lindblad_dissipator
