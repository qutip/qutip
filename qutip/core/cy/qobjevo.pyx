#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

import numpy as np
import numbers
import itertools
from functools import partial

import qutip
from .. import Qobj
from .. import data as _data
from ..coefficient import (coefficient, CompilationOptions,
                           _read_callable_signature)
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
    A class for representing time-dependent quantum objects, such as quantum
    operators and states.

    Importantly, :obj:`~QobjEvo` instances are used to represent such
    time-dependent quantum objects when working with QuTiP solvers.

    A :obj:`~QobjEvo` instance may be constructed from one of the following:

      * a callable ``f(t: double, args: dict) -> Qobj`` that returns the
        value of the quantum object at time ``t``.

      * a ``[Qobj, Coefficient]`` pair, where :obj:`~Coefficient` may also be
        any item that can be used to construct a coefficient (e.g. a function,
        a numpy array of coefficient values, a string expression).

      * a :obj:`~Qobj` (which creates a constant :obj:`~QobjEvo` term).

      * a list of such callables, pairs or :obj:`~Qobj`s.

      * a :obj:`~QobjEvo` (in which case a copy is created, all other arguments
        are ignored except ``args`` which, if passed, replaces the existing
        arguments).

    Parameters
    ----------
    Q_object : callable, list or Qobj
        A specification of the time-depedent quantum object. See the
        paragraph above for a full description and the examples section below
        for examples.

    args : dict, optional
        A dictionary that contains the arguments for the coefficients.
        Arguments may be omitted if no function or string coefficients that
        require arguments are present.

    tlist : array-like, optional
        A list of times corresponding to the values of the coefficients
        supplied as numpy arrays. If no coefficients are supplied as numpy
        arrays, ``tlist`` may be omitted, otherwise it is required.

        The times in ``tlist`` do not need to be equidistant, but must
        be sorted.

        By default, a cubic spline interpolation will be used to interpolate
        the value of the (numpy array) coefficients at time ``t``. If the
        coefficients are to be treated as step functions, pass the argument
        ``step_interpolation=True`` (see below).

    step_interpolation : bool, default=False
        By default, a cubic spline interpolation will be used to interpolate
        the value of the (numpy array) coefficients at time ``t``.
        Pass ``True`` to use step interpolation instead.

    copy : bool, default=True
        Whether to make a copy of the :obj:`Qobj` instances supplied in
        the ``Q_object`` parameter.

    compress : bool, default=True
        Whether to compress the :obj:`QobjEvo` instance terms after the
        instance has been created.

        This sums the constant terms in a single term and combines
        ``[Qobj, coefficient]`` pairs with the same :obj:`~Qobj` into a single
        pair containing the sum of the coefficients.

        See :meth:`compress`.

    Attributes
    ----------
    dims : list
        List of dimensions keeping track of the tensor structure.

    shape : (int, int)
        List of dimensions keeping track of the tensor structure.

    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.

    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form) or 'choi' (Choi matrix with tr = dimension).

    Property
    --------
    num_elements
        Number of parts composing the system.

    const:
        Does the system change depending on `t`.

    isoper:
        Indicates if the system represents an operator.

    issuper:
        Indicates if the system represents an superoperator.

    Examples
    --------

    A :obj:`~QobjEvo` constructed from a function:

    ```
    def f(t, args):
        return qutip.qeye(N) * np.exp(args['w'] * t)

    QobjEvo(f, args={'w': 1j})
    ```

    For list based :obj:`~QobjEvo`, the list must consist of :obj`~Qobj` or
    ``[Qobj, Coefficient]`` pairs:

    ```
    QobjEvo([H0, [H1, coeff1], [H2, coeff2]], args=args)
    ```

    The coefficients may be specified either using a :obj:`~Coefficient`
    object or by a function, string, numpy array or any object that
    can be passed to the :func:`~coefficient` function. See the documentation
    of :func:`coefficient` for a full description.

    An example of a coefficient specified by a function:

    ```
    def f1_t(t, args):
        return np.exp(-1j * t * args["w1"])

    QobjEvo([[H1, f1_t]], args={"w1": 1.})
    ```

    And of coefficients specified by string expressions:

    ```
    H = QobjEvo(
        [H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
        args={"w1": 1., "w2": 2.}
    )
    ```

    Coefficients maybe also be expressed as numpy arrays giving a list
    of the coefficient values:

    ```
    tlist = np.logspace(-5, 0, 100)
    H = QobjEvo(
        [H0, [H1, np.exp(-1j * tlist)], [H2, np.cos(2. * tlist)]],
        tlist=tlist
    )
    ```

    The coeffients array must have the same len as the tlist.

    A :obj:`~QobjEvo` may also be built using simple arithmetic operations
    combining :obj:`~Qobj` with :obj:`~Coefficient`, for example:

    ```
    coeff = qutip.coefficient("exp(-1j*w1*t)", args={"w1": 1})
    qevo = H0 + H1 * coeff
    ```
    """
    def __init__(QobjEvo self, Q_object, args=None, tlist=None,
                 step_interpolation=False, copy=True, compress=True):
        if isinstance(Q_object, QobjEvo):
            self.dims = Q_object.dims.copy()
            self.shape = Q_object.shape
            self._shift_dt = (<QobjEvo> Q_object)._shift_dt
            self._issuper = (<QobjEvo> Q_object)._issuper
            self._isoper = (<QobjEvo> Q_object)._isoper
            self.elements = (<QobjEvo> Q_object).elements.copy()
            if args:
                self.arguments(args)
            if compress:
                self.compress()
            return

        self.elements = []
        self.dims = None
        self.shape = (0, 0)
        self._issuper = -1
        self._isoper = -1
        self._shift_dt = 0
        args = args or {}

        if (
            isinstance(Q_object, list)
            and len(Q_object) == 2
            and isinstance(Q_object[0], Qobj)
            and not isinstance(Q_object[1], (Qobj, list))
        ):
            # The format is [Qobj, coefficient]
            Q_object = [Q_object]

        if isinstance(Q_object, list):
            for op in Q_object:
                self.elements.append(
                    self._read_element(op, copy, tlist, args, step_interpolation)
                )
        else:
            self.elements.append(
              self._read_element(Q_object, copy, tlist, args, step_interpolation)
            )

        if compress:
            self.compress()

    def _read_element(self, op, copy, tlist, args, use_step_func):
        """ Read a Q_object item and return an element for that item. """
        if isinstance(op, Qobj):
            out = _ConstantElement(op.copy() if copy else op)
            qobj = op
        elif isinstance(op, list):
            out = _EvoElement(
                op[0].copy() if copy else op[0],
                coefficient(op[1], tlist=tlist, args=args,
                            _stepInterpolation=use_step_func)
            )
            qobj = op[0]
        elif callable(op):
            use_kw, args, kwargs = _read_callable_signature(op, args)
            if use_kw:
                out = _KwFuncElement(op, args, kwargs)
            else:
                out = _FuncElement(op, args)
            qobj = out.qobj(0)
            if not isinstance(qobj, Qobj):
                raise TypeError(
                    "Function based time-dependent elements must have the"
                    " signature f(t: double, args: dict) -> Qobj, but"
                    " {!r} returned: {!r}".format(op, qobj)
                )

        else:
            raise TypeError(
                "QobjEvo terms should be Qobjs, a list of [Qobj, coefficient],"
                " or a function f(t: double, args: dict) -> Qobj, but"
                " received: {!r}".format(op)
            )

        if self.dims is None:
            self.dims = qobj.dims
            self.shape = qobj.shape
            self.type = qobj.type
            self.superrep = qobj.superrep
        elif self.dims != qobj.dims or self.shape != qobj.shape:
            raise ValueError(
                f"QobjEvo term {op!r} has dims {qobj.dims!r} and shape"
                f" {qobj.shape!r} but previous terms had dims {self.dims!r}"
                f" and shape {self.shape!r}."
            )
        elif self.superrep != qobj.superrep:
            raise ValueError(
                f"QobjEvo term {op!r} has superrep {qobj.superrep!r} but "
                f"previous terms had superrep {self.superrep!r}."
            )

        return out

    def __call__(self, double t, dict args=None):
        if args:
            return QobjEvo(self, args=args)(t)
        return Qobj(
            self._call(t), dims=self.dims, copy=False,
            type=self.type, superrep=self.superrep
        )

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
        return QobjEvo(self, compress=False)

    def arguments(QobjEvo self, dict new_args):
        """Update the arguments"""
        cache = []
        self.elements = [
            element.replace_arguments(new_args, cache=cache)
            for element in self.elements
        ]


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
        Look for redundance in the QobjEvo components:

        Constant parts, (:class:`~Qobj` without :class:`~Coefficient`) will be
        summed.
        Pairs ``[Qobj, Coefficient]`` with the same :class:`~Qobj` are merged.

        Example:
        ``[[sigmax(), f1], [sigmax(), f2]]``` -> ``[[sigmax(), f1+f2]]``

        The :class:`~QobjEvo` is transformed inplace.

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

    cpdef double complex expect_data(QobjEvo self, double t, Data state) except *:
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

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state) except *:
        """For Dense state, `column_stack_dense` can be done inplace if in
        fortran format."""
        cdef size_t nrow = state.shape[0]
        cdef _BaseElement part
        cdef double complex out = 0., coeff
        cdef Data part_data
        t = self._prepare(t, state)
        if self.issuper:
            if state.shape[1] != 1:
                state = column_stack_dense(state, inplace=state.fortran)
            try:
                for element in self.elements:
                    part = (<_BaseElement> element)
                    coeff = part.coeff(t)
                    part_data = part.data(t)
                    out += coeff * expect_super_data_dense(part_data, state)
            finally:
                if state.fortran:
                    # `state` was reshaped inplace, restore it's original shape
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
