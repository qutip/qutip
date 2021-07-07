#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

import numpy as np
import numbers
import itertools
from functools import partial

import qutip
from .. import Qobj
from .. import data as _data
from ..coefficient import coefficient, CompilationOptions
from ._element import *
from ..dimensions import type_from_dims
from qutip.settings import settings

from qutip.core.cy._element cimport _BaseElement
from qutip.core.data cimport Dense, Data, dense
from qutip.core.data.expect cimport *
from qutip.core.data.reshape cimport (column_stack_dense, column_unstack_dense)
from qutip.core.cy.coefficient cimport Coefficient
from libc.math cimport fabs

__all__ = ['QobjEvo']

cdef class QobjEvo:
    """
    A class for representing time-dependent quantum objects,
    such as quantum operators and states.

    The :obj:`QobjEvo` class is a representation of time-dependent Qutip quantum
    objects (:obj:`Qobj`) for system accepted by solvers. The QobjEvo
    are constructed from a nested list of Qobj with their time-dependent
    coefficients, or for function returning a Qobj.

    For a QobjEvo based on a function, the function signature must be:
        ``f(t: double, args: dict) -> Qobj``.

    *Examples*
    ```
    def f(t, args):
        return qutip.qeye(N) * np.exp(args['w'] * t)

    QobjEvo(f, args={'w': 1j})
    ```

    For list based QobjEvo, the list must be comprised of ``Qobj`` and pair
    ``[Qobj, coefficient]``.
    *Examples*
    ```
    QobjEvo([H0, [H1, coeff1], [H2, coeff2]], args=args)
    ```

    The time-dependent coefficients are either functions, strings, numpy arrays
    or :obj:``Cubic_Spline``. For function format, the function signature
    must be f(t, args).
    *Examples*
    ```
    def f1_t(t, args):
        return np.exp(-1j * t * args["w1"])

    QobjEvo([[H1, f1_t]], args={"w1":1.})
    ```

    With string based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:
        ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``pi``,
        ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``,
        ``exp``, ``log``, ``log10``, ``erf``, ``zerf``, ``sqrt``,
        ``real``, ``imag``, ``conj``, ``abs``, ``norm``, ``arg``, ``proj``,
        numpy as ``np``, scipy.special as ``spe`` and
        ``cython_special`` (cython interface).
    *Examples*
    ```
    H = QobjEvo([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})
    ```

    With numpy array, the array must be an 1d of dtype float or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    By default, a cubic spline interpolation will be used for the coefficient
    at time t. If the coefficients are to be treated as step function, use the
    keyword `step_interpolation=True`.
    *Examples*
    ```
    tlist = np.logspace(-5,0,100)
    H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)
    ```

    With `qutip.Cubic_Spline` are also valid coefficient.

    See qutip.coefficient

    `args` is a dict of (name:object).
    The name must be a valid variables string.

    QobjEvo can also be built with the product of `Qobj` with `Coefficient`.
    *Examples*
    ```
    coeff = qutip.coefficient("exp(-1j*w1*t)", args={"w1":1})
    qevo = H0 + H1 * coeff
    ```

    Parameters
    ----------
    Q_object : array_like
        Data for vector/matrix representation of the quantum object.

    args : dict
        dictionary that contain the arguments for the coefficients

    tlist : array_like
        List of times corresponding to the values of the numpy-array
        coefficients are applied.

    copy : bool
        Make a copy of the Qobj composing the QobjEvo.

    step_interpolation : bool
        For array :obj:`Coefficient`, use step interpolation instead of spline.

    Attributes
    ----------
    dims : list
        List of dimensions keeping track of the tensor structure.

    shape : (int, int)
        List of dimensions keeping track of the tensor structure.

    Property
    --------
    num_obj
        Number of parts composing the system.

    const:
        Does the system change depending on `t`.

    isoper:
        Indicates if the system represents an operator.

    issuper:
        Indicates if the system represents an superoperator.
    """
    def __init__(QobjEvo self, Q_object, args=None, tlist=None,
                 step_interpolation=False, copy=True):
        if isinstance(Q_object, QobjEvo):
            self.dims = Q_object.dims.copy()
            self.shape = Q_object.shape
            self._shift_dt = (<QobjEvo> Q_object)._shift_dt
            self._issuper = (<QobjEvo> Q_object)._issuper
            self._isoper = (<QobjEvo> Q_object)._isoper
            self.elements = (<QobjEvo> Q_object).elements.copy()
            if args:
                self.arguments(args)
            return

        self.elements = []
        self.dims = None
        self.shape = (0, 0)
        self._issuper = -1
        self._isoper = -1
        self._shift_dt = 0
        args = args or {}

        use_step_func = args.get("_step_func_coeff", 0) or step_interpolation

        if (
            isinstance(Q_object, list)
            and len(Q_object) == 2
            and isinstance(Q_object[0], Qobj)
            and not isinstance(Q_object[1], (Qobj, list))
        ):
            # The format is [Qobj, f/str]
            Q_object = [Q_object]

        if isinstance(Q_object, Qobj):
            self.elements = [
                _ConstantElement(Q_object.copy() if copy else Q_object)
            ]
            self.dims = Q_object.dims
            self.shape = Q_object.shape

        elif isinstance(Q_object, list):
            for op in Q_object:
                self.elements.append(
                    self._read_element(op, copy, tlist, args, use_step_func)
                )
            self.compress()

        elif callable(Q_object):
            qobj = Q_object(0, args)
            if not isinstance(qobj, Qobj):
                raise ValueError("Function based time-dependent system must "
                                 "have the signature "
                                 "`f(t: double, args: dict) -> Qobj`")
            self.dims = qobj.dims
            self.shape = qobj.shape
            self.elements.append(_FuncElement(Q_object, args))

        else:
            raise TypeError("Format not understood")

    def _read_element(self, op, copy, tlist, args, use_step_func):
        """ Read one value of the list format."""
        if isinstance(op, Qobj):
            out = _ConstantElement(op.copy() if copy else op)
            _dims = op.dims
            _shape = op.shape
        elif isinstance(op, list):
            out = _EvoElement(
                op[0].copy() if copy else op[0],
                coefficient(op[1], tlist=tlist, args=args,
                            _stepInterpolation=use_step_func)
            )
            _dims = op[0].dims
            _shape = op[0].shape
        else:
            raise TypeError("List QobjEvo should be comprised of Qobj and"
                            " list of `[Qobj, coefficient]`")

        if self.dims is None:
            self.dims = _dims
            self.shape = _shape
        else:
            if self.dims != _dims:
                raise ValueError("incompatible dimensions " +
                                 str(self.dims) + ", " + str(_dims))
        return out

    def __call__(self, double t, dict args=None):
        if args:
            return QobjEvo(self, args=args)(t)
        return Qobj(self._call(t), dims=self.dims, copy=False)

    cpdef Data _call(QobjEvo self, double t):
        t = self._prepare(t, None)
        cdef Data out
        cdef _BaseElement part = self.elements[0]
        out = _data.mul(part.data(t),
                        part.coeff(t))
        for element in self.elements[1:]:
            part = <_BaseElement> element
            out = _data.add(
                out,
                part.data(t),
                part.coeff(t)
            )
        return out

    cdef double _prepare(QobjEvo self, double t, Data state=None):
        """ Precomputation before computing getting the element at `t`"""
        return t + self._shift_dt

    def copy(QobjEvo self):
        """Return a copy of this `QobjEvo`"""
        return QobjEvo(self)

    def arguments(QobjEvo self, dict new_args):
        """Update the arguments"""
        safe = [] # storage for _FuncElement's instance management.
        self.elements = [element.replace_arguments(new_args, safe)
                         for element in self.elements]


    ###########################################################################
    # Math function                                                           #
    ###########################################################################
    def __add__(left, right):
        if isinstance(left, QobjEvo):
            self = left
            other = right
        else:
            self = right
            other = left
        if not isinstance(other, (Qobj, QobjEvo, numbers.Number)):
            return NotImplemented
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        if not isinstance(other, (Qobj, QobjEvo, numbers.Number)):
            return NotImplemented
        res = self.copy()
        res += other
        return res

    def __iadd__(QobjEvo self, other):
        cdef _BaseElement element
        if isinstance(other, QobjEvo):
            if other.dims != self.dims:
                raise TypeError("incompatible dimensions" +
                                 str(self.dims) + ", " + str(other.dims))
            for element in (<QobjEvo> other).elements:
                self.elements.append(element)
        elif isinstance(other, Qobj):
            if other.dims != self.dims:
                raise TypeError("incompatible dimensions" +
                                 str(self.dims) + ", " + str(other.dims))
            self.elements.append(_ConstantElement(other))
        elif (
            isinstance(other, numbers.Number) and
            self.dims[0] == self.dims[1]
        ):
            self.elements.append(_ConstantElement(other * qutip.qeye(self.dims[0])))
        else:
            return NotImplemented
        return self

    def __sub__(left, right):
        if isinstance(left, QobjEvo):
            res = left.copy()
            res += -right
            return res
        else:
            res = -right.copy()
            res += left
            return res

    def __rsub__(self, other):
        if not isinstance(other, (Qobj, QobjEvo, numbers.Number)):
            return NotImplemented
        res = -self
        res += other
        return res

    def __isub__(self, other):
        if not isinstance(other, (Qobj, QobjEvo, numbers.Number)):
            return NotImplemented
        self += (-other)
        return self

    def __matmul__(left, right):
        cdef QobjEvo res
        if isinstance(left, QobjEvo):
            return left.copy().__imatmul__(right)
        elif isinstance(left, Qobj):
            if left.dims[1] != (<QobjEvo> right).dims[0]:
                raise TypeError("incompatible dimensions" +
                                 str(left.dims[1]) + ", " +
                                 str((<QobjEvo> right).dims[0]))
            res = right.copy()
            res.dims = [left.dims[0], right.dims[1]]
            res.shape = (left.shape[0], right.shape[1])
            left = _ConstantElement(left)
            res.elements = [left @ element for element in res.elements]
            res._issuper = -1
            res._isoper = -1
            return res
        else:
            return NotImplemented

    def __rmatmul__(QobjEvo self, other):
        if isinstance(other, Qobj):
            if other.dims[1] != self.dims[0]:
                raise TypeError("incompatible dimensions" +
                                 str(other.dims[1]) + ", " +
                                 str(self.dims[0]))
            res = self.copy()
            res.dims = [other.dims[0], res.dims[1]]
            res.shape = (other.shape[0], res.shape[1])
            other = _ConstantElement(other)
            res.elements = [other @ element for element in res.elements]
            res._issuper = -1
            res._isoper = -1
            return res
        else:
            return NotImplemented

    def __imatmul__(QobjEvo self, other):
        if isinstance(other, (Qobj, QobjEvo)):
            if self.dims[1] != other.dims[0]:
                raise TypeError("incompatible dimensions" +
                                str(self.dims[1]) + ", " +
                                str(other.dims[0]))
            self.dims = [self.dims[0], other.dims[1]]
            self.shape = (self.shape[0], other.shape[1])
            self._issuper = -1
            self._isoper = -1
            if isinstance(other, Qobj):
                other = _ConstantElement(other)
                self.elements = [element @ other for element in self.elements]

            elif isinstance(other, QobjEvo):
                self.elements = [left @ right
                    for left, right in itertools.product(
                        self.elements, (<QobjEvo> other).elements
                    )]
        else:
            return NotImplemented
        return self

    def __mul__(left, right):
        if isinstance(left, QobjEvo):
            return left.copy().__imul__(right)
        elif isinstance(left, Qobj):
            return right.__rmatmul__(left)
        elif isinstance(left, (numbers.Number, Coefficient)):
            return right.copy().__imul__(left)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Qobj):
            return self.__rmatmul__(other)
        else:
            res = self.copy()
            res *= other
            return res

    def __imul__(QobjEvo self, other):
        if isinstance(other, (Qobj, QobjEvo)):
            self @= other
        elif isinstance(other, numbers.Number):
            self.elements = [element * other for element in self.elements]
        elif isinstance(other, Coefficient):
            other = _EvoElement(qutip.qeye(self.dims[1]), other)
            self.elements = [element @ other for element in self.elements]
        else:
            return NotImplemented
        return self

    def __truediv__(left, right):
        if isinstance(left, QobjEvo) and isinstance(right, numbers.Number):
            res = left.copy()
            res *= 1 / right
            return res
        return NotImplemented

    def __idiv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        self *= 1 / other
        return self

    def __neg__(self):
        res = self.copy()
        res *= -1
        return res

    ###########################################################################
    # tensor                                                                  #
    ###########################################################################
    def __and__(left, right):
        """
        Syntax shortcut for tensor:
        A & B ==> tensor(A, B)
        """
        return qutip.tensor(left, right)

    ###########################################################################
    # Unary transformation                                                    #
    ###########################################################################
    def trans(self):
        """ Transpose of the quantum object """
        cdef QobjEvo res = self.copy()
        res.elements = [element.linear_map(Qobj.trans)
                        for element in res.elements]
        return res

    def conj(self):
        """Get the element-wise conjugation of the quantum object."""
        cdef QobjEvo res = self.copy()
        res.elements = [element.linear_map(Qobj.conj, True)
                        for element in res.elements]
        return res

    def dag(self):
        """Get the Hermitian adjoint of the quantum object."""
        cdef QobjEvo res = self.copy()
        res.elements = [element.linear_map(Qobj.dag, True)
                        for element in res.elements]
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
        return self.linear_map(partial(Qobj.to, data_type=data_type),
                               _skip_check=True)

    def _insert_time_shift(QobjEvo self, dt):
        """
        Add a shift in the time `t = t + _t0`.
        To be used in correlation.py only. It does not propage safely with
        binop between QobjEvo with different shift.
        """
        cdef QobjEvo out = self.copy()
        out._shift_dt = dt
        return out

    def tidyup(self, atol=1e-12):
        """Removes small elements from quantum object."""
        for element in self.elements:
            if type(element) is _ConstantElement:
                element = _ConstantElement(element.qobj(0).tidyup(atol))
            elif type(element) is _EvoElement:
                element = _EvoElement(element.qobj(0).tidyup(atol),
                                      element._coefficient)
        return self

    def linear_map(self, op_mapping, *, _skip_check=False):
        """
        Apply mapping to each Qobj contribution.

        Example:
        `QobjEvo([sigmax(), coeff]).linear_map(spre)`
        gives the same result has
        `QobjEvo([spre(sigmax()), coeff])`

        Returns
        -------
        :class:`.QobjEvo`
            Modified object

        Notes
        -----
        Does not modify the coefficients, thus `linear_map(conj)` would not
        give the the conjugate of the QobjEvo. It's only valid for linear
        transformations.
        """
        if not _skip_check:
            out = op_mapping(self(0))
            if not isinstance(out, Qobj):
                raise TypeError("The op_mapping function must return a Qobj")
        cdef QobjEvo res = self.copy()
        res.elements = [element.linear_map(op_mapping) for element in res.elements]
        res.dims = res.elements[0].qobj(0).dims
        res.shape = res.elements[0].qobj(0).shape
        res._issuper = res.elements[0].qobj(0).issuper
        res._isoper = res.elements[0].qobj(0).isoper
        if not _skip_check:
            if res(0) != out:
                raise ValueError("The mapping is not linear")

        return res

    ###########################################################################
    # Cleaning and compress                                                   #
    ###########################################################################
    def _compress_merge_qobj(self, coeff_elements):
        "merge element with matching qobj: [A, f1], [A, f2] -> [A, f1+f2]"
        cleaned_elements = []
        # Mimic a dict with Qobj not hashable
        qobjs = []
        coeffs = []
        for element in coeff_elements:
            for i, qobj in enumerate(qobjs):
                if element.qobj(0) == qobj:
                    coeffs[i] = coeffs[i] + element._coefficient
                    break
            else:
                qobjs.append(element.qobj(0))
                coeffs.append(element._coefficient)
        for qobj, coeff in zip(qobjs, coeffs):
            cleaned_elements.append(_EvoElement(qobj, coeff))
        return cleaned_elements

    def compress(self):
        """
        Look for redundance in the QobjEvo components, then merge them.

        Example:
        `[[sigmax(), f1], [sigmax(), f2]]` -> `[[sigmax(), f1+f2]]`

        The `QobjEvo` is transformed inplace.

        Returns
        -------
        None
        """
        cte_elements = []
        coeff_elements = []
        func_elements = []
        for element in self.elements:
            if type(element) is _ConstantElement:
                cte_elements.append(element)
            elif type(element) is _EvoElement:
                coeff_elements.append(element)
            else:
                func_elements.append(element)

        cleaned_elements = []
        if len(cte_elements) >= 2:
            # Multiple constant parts
            cleaned_elements.append(_ConstantElement(
                sum(element.qobj(0) for element in cte_elements)))
        else:
            cleaned_elements += cte_elements

        coeff_elements = self._compress_merge_qobj(coeff_elements)
        cleaned_elements += coeff_elements + func_elements

        self.elements = cleaned_elements

    ###########################################################################
    # properties                                                              #
    ###########################################################################
    @property
    def num_elements(self):
        """Number of parts composing the system"""
        return len(self.elements)

    @property
    def isconstant(self):
        """Does the system change depending on `t`"""
        return not any(type(element) is not _ConstantElement for element in self.elements)

    @property
    def isoper(self):
        """Indicates if the system represents an operator."""
        # TODO: isoper should be part of dims
        if self._isoper == -1:
            self._isoper = type_from_dims(self.dims) == "oper"
        return self._isoper

    @property
    def issuper(self):
        """Indicates if the system represents a superoperator."""
        # TODO: issuper should/will be part of dims
        # remove self._issuper then
        if self._issuper == -1:
            self._issuper = type_from_dims(self.dims) == "super"
        return self._issuper

    ###########################################################################
    # operation methods                                                       #
    ###########################################################################
    def expect(QobjEvo self, double t, state):
        """
        Expectation value of this operator at time `t` with the state.

        Parameters
        ----------
        t : float
            Time of the operator to apply.
        state : Qobj
            right matrix of the product

        Returns
        -------
        expect : float or complex
            `state.adjoint() @ self @ state` if `state` is a ket.
            `trace(self @ matrix)` is `state` is an operator or operator-ket.
        """
        # TODO: remove reading from `settings` for a typed value when options
        # support property.
        cdef float herm_rtol = settings.core['rtol']
        if not isinstance(state, Qobj):
            raise TypeError("A Qobj state is expected")
        if not (self.isoper or self.issuper):
            raise ValueError("Must be an operator or super operator to compute"
                             " an expectation value")
        if not (
            (self.dims[1] == state.dims[0]) or
            (self.issuper and self.dims[1] == state.dims)
        ):
            raise ValueError("incompatible dimensions " + str(self.dims) +
                             ", " + str(state.dims))
        out = self.expect_data(t, state.data)
        if out == 0 or (out.real and fabs(out.imag / out.real) < herm_rtol):
            return out.real
        return out

    cpdef double complex expect_data(QobjEvo self, double t, Data state):
        """
        Expectation is defined as `state.adjoint() @ self @ state` if
        `state` is a vector, or `state` is an operator and `self` is a
        superoperator.  If `state` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        if type(state) is Dense:
            return self._expect_dense(t, state)
        cdef _BaseElement part
        cdef double complex out = 0., coeff
        cdef Data part_data
        cdef object expect_func
        t = self._prepare(t, state)
        if self.issuper:
            if state.shape[1] != 1:
                state = _data.column_stack(state)
            expect_func = _data.expect_super
        else:
            expect_func = _data.expect

        for element in self.elements:
            part = (<_BaseElement> element)
            coeff = part.coeff(t)
            part_data = part.data(t)
            out += coeff * expect_func(part_data, state)
        return out

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state):
        "For Dense state, `column_stack_dense` can be done inplace."
        cdef size_t nrow = state.shape[0]
        cdef _BaseElement part
        cdef double complex out = 0., coeff
        cdef Data part_data
        t = self._prepare(t, state)
        if self.issuper:
            if state.shape[1] != 1:
                state = column_stack_dense(state, inplace=state.fortran)
            for element in self.elements:
                part = (<_BaseElement> element)
                coeff = part.coeff(t)
                part_data = part.data(t)
                out += coeff * expect_super_data_dense(part_data, state)
            if state.fortran:
                column_unstack_dense(state, nrow, inplace=state.fortran)
        else:
            for element in self.elements:
                part = (<_BaseElement> element)
                coeff = part.coeff(t)
                part_data = part.data(t)
                out += coeff * expect_data_dense(part_data, state)
        return out

    def matmul(self, double t, state):
        """
        Product of this operator at time `t` to the state.
        `self(t) @ state`

        Parameters
        ----------
        t : float
            Time of the operator to apply.
        state : Qobj
            right matrix of the product

        Returns
        -------
        product : Qobj
            The result product as a Qobj
        """
        if not isinstance(state, Qobj):
            raise TypeError("A Qobj state is expected")

        if self.dims[1] != state.dims[0]:
            raise ValueError("incompatible dimensions " + str(self.dims[1]) +
                             ", " + str(state.dims[0]))

        return Qobj(self.matmul_data(t, state.data),
                    dims=[self.dims[0],state.dims[1]],
                    copy=False
                   )

    cpdef Data matmul_data(QobjEvo self, double t, Data state, Data out=None):
        """out += self(t) @ state"""
        cdef _BaseElement part
        t = self._prepare(t, state)
        if out is None and type(state) is Dense:
            out = dense.zeros(self.shape[0], state.shape[1],
                              (<Dense> state).fortran)
        elif out is None:
            out = _data.zeros[type(state)](self.shape[0], state.shape[1])

        for element in self.elements:
            part = (<_BaseElement> element)
            out = part.matmul_data_t(t, state, out)
        return out
