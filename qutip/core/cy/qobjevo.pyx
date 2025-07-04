#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

import numpy as np
import numbers
import itertools
from functools import partial

import qutip
from .. import Qobj
from .. import data as _data
from ..dimensions import Dimensions
from ..coefficient import coefficient, CompilationOptions
from ._element import *
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

    Importantly, :obj:`.QobjEvo` instances are used to represent such
    time-dependent quantum objects when working with QuTiP solvers.

    A :obj:`.QobjEvo` instance may be constructed from one of the following:

      * a callable ``f(t: double, args: dict) -> Qobj`` that returns the
        value of the quantum object at time ``t``.

      * a ``[Qobj, Coefficient]`` pair, where the :obj:`Coefficient` may be any
        item that :func:`.coefficient` can accept (e.g. a function, a numpy
        array of coefficient values, a string expression).

      * a :obj:`.Qobj` (which creates a constant :obj:`.QobjEvo` term).

      * a list of such callables, pairs or :obj:`.Qobj`\s.

      * a :obj:`.QobjEvo` (in which case a copy is created, all other arguments
        are ignored except ``args`` which, if passed, replaces the existing
        arguments).

    Parameters
    ----------
    Q_object : callable, list or :obj:`.Qobj`
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
        ``order=0`` (see below).

    order : int, default=3
        Order of the spline interpolation that is to be used to interpolate
        the value of the (numpy array) coefficients at time ``t``.
        ``0`` use previous or left value.

    copy : bool, default=True
        Whether to make a copy of the :obj:`.Qobj` instances supplied in
        the ``Q_object`` parameter.

    compress : bool, default=True
        Whether to compress the :obj:`.QobjEvo` instance terms after the
        instance has been created.

        This sums the constant terms in a single term and combines
        ``[Qobj, coefficient]`` pairs with the same :obj:`.Qobj` into a single
        pair containing the sum of the coefficients.

        See :meth:`compress`.

    function_style : {None, "pythonic", "dict", "auto"}
        The style of function signature used by callables in ``Q_object``.
        If style is ``None``, the value of
        ``qutip.settings.core["function_coefficient_style"]``
        is used. Otherwise the supplied value overrides the global setting.


    boundary_conditions : 2-Tuple, str or None, optional
        Boundary conditions for spline evaluation. Default value is `None`.
        Correspond to `bc_type` of scipy.interpolate.make_interp_spline.
        Refer to Scipy's documentation for further details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html

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

    Examples
    --------

    A :obj:`.QobjEvo` constructed from a function:

    .. code-block::

        def f(t, args):
            return qutip.qeye(N) * np.exp(args['w'] * t)

        QobjEvo(f, args={'w': 1j})


    For list based :obj:`.QobjEvo`, the list must consist of :obj:`.Qobj` or
    ``[Qobj, Coefficient]`` pairs:

    .. code-block::

        QobjEvo([H0, [H1, coeff1], [H2, coeff2]], args=args)

    The coefficients may be specified either using a :obj:`Coefficient` object
    or by a function, string, numpy array or any object that can be passed to
    the :func:`.coefficient` function. See the documentation of
    :func:`.coefficient` for a full description.

    An example of a coefficient specified by a function:

    .. code-block::

        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        QobjEvo([[H1, f1_t]], args={"w1": 1.})

    And of coefficients specified by string expressions:

    .. code-block::

        H = QobjEvo(
            [H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
            args={"w1": 1., "w2": 2.}
        )

    Coefficients maybe also be expressed as numpy arrays giving a list
    of the coefficient values:

    .. code-block:: python

        tlist = np.logspace(-5, 0, 100)
        H = QobjEvo(
            [H0, [H1, np.exp(-1j * tlist)], [H2, np.cos(2. * tlist)]],
            tlist=tlist
        )

    The coeffients array must have the same len as the tlist.

    A :obj:`.QobjEvo` may also be built using simple arithmetic operations
    combining :obj:`.Qobj` with :obj:`Coefficient`, for example:

    .. code-block:: python

        coeff = qutip.coefficient("exp(-1j*w1*t)", args={"w1": 1})
        qevo = H0 + H1 * coeff

    """
    def __init__(QobjEvo self, Q_object, args=None, *, copy=True, compress=True,
                 function_style=None,
                 tlist=None, order=3, boundary_conditions=None):
        if isinstance(Q_object, QobjEvo):
            self._dims = Q_object._dims
            self.shape = Q_object.shape
            self.elements = (<QobjEvo> Q_object).elements.copy()
            self._feedback_functions = Q_object._feedback_functions.copy()
            self._solver_only_feedback = Q_object._solver_only_feedback.copy()
            if args:
                self.arguments(args)
            if compress:
                self.compress()
            return

        self.elements = []
        self._dims = None
        self.shape = (0, 0)
        self._feedback_functions = {}
        self._solver_only_feedback = {}
        args = self._read_args(args or {})

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
                    self._read_element(
                        op, copy=copy, tlist=tlist, args=args, order=order,
                        function_style=function_style,
                        boundary_conditions=boundary_conditions
                    )
                )
        else:
            self.elements.append(
                self._read_element(
                    Q_object, copy=copy, tlist=tlist, args=args, order=order,
                    function_style=function_style,
                    boundary_conditions=boundary_conditions
                )
            )

        for key in self._feedback_functions:
            # During _read_args, the dims could not have been set yet.
            # To set the dims, for function QobjEvo, they need to be called.
            # But to be called, the feedback args need to be read...
            self._feedback_functions[key].check_consistency(self._dims)

        if compress:
            self.compress()

    def __repr__(self):
        cls = self.__class__.__name__
        repr_str = f'{cls}: dims = {self.dims}, shape = {self.shape}, '
        repr_str += f'type = {self.type}, superrep = {self.superrep}, '
        repr_str += f'isconstant = {self.isconstant}, '
        repr_str += f'num_elements = {self.num_elements}'
        feedback_pairs = []
        for key, val in self._feedback_functions.items():
            feedback_pairs.append(key + ":" + repr(val))
        for key, val in self._solver_only_feedback.items():
            feedback_pairs.append(key + ":" + val)
        if feedback_pairs:
            repr_str += f', feedback = {feedback_pairs}'

        return repr_str

    def _read_element(self, op, copy, tlist, args, order, function_style,
                      boundary_conditions):
        """ Read a Q_object item and return an element for that item. """
        if isinstance(op, Qobj):
            out = _ConstantElement(op.copy() if copy else op)
            qobj = op
        elif isinstance(op, list):
            out = _EvoElement(
                op[0].copy() if copy else op[0],
                coefficient(op[1], tlist=tlist, args=args, order=order,
                            boundary_conditions=boundary_conditions)
            )
            qobj = op[0]
        elif isinstance(op, _BaseElement):
            out = op
            qobj = op.qobj(0)
        elif callable(op):
            out = _FuncElement(op, args, style=function_style)
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

        if self._dims is None:
            self._dims = qobj._dims
            self.shape = qobj.shape
        elif self._dims != qobj._dims:
            raise ValueError(
                f"QobjEvo term {op!r} has dims {qobj.dims!r} and shape"
                f" {qobj.shape!r} but previous terms had dims {self.dims!r}"
                f" and shape {self.shape!r}."
            )

        return out

    @classmethod
    def _restore(cls, elements, dims, shape):
        """Recreate a QobjEvo without using __init__. """
        cdef QobjEvo out = cls.__new__(cls)
        out.elements = elements
        out._dims = dims
        out.shape = shape
        return out

    def _getstate(self):
        """ Obtain the state """
        # For jax pytree representation
        # auto_pickle create similar method __getstate__, but since it's
        # automatically created, it could change depending on cython version
        # etc., so we create our own.
        return {
            "elements": self.elements,
            "dims": self._dims,
            "shape": self.shape,
        }

    def __call__(self, double t, dict _args=None, **kwargs):
        """
        Get the :obj:`.Qobj` at ``t``.

        Parameters
        ----------
        t : float
            Time at which the :obj:`.QobjEvo` is to be evalued.

        _args : dict [optional]
            New arguments as a dict. Update args with ``arguments(new_args)``.

        **kwargs :
            New arguments as a keywors. Update args with
            ``arguments(**new_args)``.

        Notes
        -----
        If both the positional ``_args`` and keywords are passed new values
        from both will be used. If a key is present with both, the
        ``_args`` dict value will take priority.
        """
        if _args is not None or kwargs:
            if _args is not None:
                kwargs.update(_args)
            return QobjEvo(self, args=kwargs)(t)

        t = self._prepare(t, None)

        if self.isconstant:
            # For constant QobjEvo's, we sum the contained Qobjs directly in
            # order to retain the cached values of attributes like .isherm when
            # possible, rather than calling _call(t) which may lose this cached
            # information.
            return sum(element.qobj(t) for element in self.elements)

        cdef _BaseElement part = self.elements[0]
        cdef double complex coeff = part.coeff(t)
        obj = part.qobj(t)
        cdef Data out = _data.mul(obj.data, coeff)
        cdef bint isherm = <bint> obj._isherm and coeff.imag == 0
        for element in self.elements[1:]:
            part = <_BaseElement> element
            coeff = part.coeff(t)
            obj = part.qobj(t)
            isherm &= <bint> obj._isherm and coeff.imag == 0
            out = _data.add(out, obj.data, coeff)

        return Qobj(out, dims=self._dims, copy=False, isherm=isherm or None)

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

    cdef object _prepare(QobjEvo self, object t, Data state=None):
        """ Precomputation before computing getting the element at `t`"""
        # We keep the function for feedback eventually
        if self._feedback_functions and state is not None:
            new_args = {
                key: func(t, state)
                for key, func in self._feedback_functions.items()
            }
            cache = []
            self.elements = [
                element.replace_arguments(new_args, cache=cache)
                for element in self.elements
            ]

        return t

    def copy(QobjEvo self):
        """Return a copy of this :obj:`.QobjEvo`"""
        return QobjEvo(self, compress=False)

    def arguments(QobjEvo self, dict _args=None, **kwargs):
        """
        Update the arguments.

        Parameters
        ----------
        _args : dict [optional]
            New arguments as a dict. Update args with ``arguments(new_args)``.

        **kwargs :
            New arguments as a keywors. Update args with
            ``arguments(**new_args)``.

        Notes
        -----
        If both the positional ``_args`` and keywords are passed new values
        from both will be used. If a key is present with both, the ``_args``
        dict value will take priority.
        """
        if _args is not None:
            kwargs.update(_args)
        cache = []
        kwargs = self._read_args(kwargs)
        self.elements = [
            element.replace_arguments(kwargs, cache=cache)
            for element in self.elements
        ]

    def _read_args(self, args):
        """
        Filter feedback args from normal args.
        """
        new_args = {}
        for key, val in args.items():
            if isinstance(val, _Feedback):
                new_args[key] = val.default
                if self._dims is not None:
                    val.check_consistency(self._dims)
                if callable(val):
                    self._feedback_functions[key] = val
                else:
                    self._solver_only_feedback[key] = val.code
            else:
                new_args[key] = val
                if key in self._feedback_functions:
                    del self._feedback_functions[key]
                if key in self._solver_only_feedback:
                    del self._solver_only_feedback[key]

        return new_args

    def _register_feedback(self, solvers_feeds, solver):
        """
        Receive feedback source from solver.

        Parameters
        ----------
        solvers_feeds : dict[str]
            When ``feedback={key: solver_specific}`` is used, update arguments
            with ``args[key] = solvers_feeds[solver_specific]``.

        solver: str
            Name of the solver for the error message.
        """
        new_args = {}
        for key, feed in self._solver_only_feedback.items():
            if feed not in solvers_feeds:
                raise ValueError(
                    f"Desired feedback {key} is not available for the {solver}."
                )
            new_args[key] = solvers_feeds[feed]

        if new_args:
            cache = []
            self.elements = [
                element.replace_arguments(new_args, cache=cache)
                for element in self.elements
            ]

    def _update_feedback(QobjEvo self, QobjEvo other=None):
        """
        Merge feedback from ``op`` into self.
        """
        if other is not None:
            if not self._feedback_functions and other._feedback_functions:
                self._feedback_functions = other._feedback_functions.copy()
            elif other._feedback_functions:
                self._feedback_functions.update(other._feedback_functions)

            if not self._solver_only_feedback and other._solver_only_feedback:
                self._solver_only_feedback = other._solver_only_feedback.copy()
            elif other._solver_only_feedback:
                self._solver_only_feedback.update(other._solver_only_feedback)

        if self._feedback_functions:
            for key in self._feedback_functions:
                self._feedback_functions[key].check_consistency(self._dims)

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
            if other._dims != self._dims:
                raise TypeError("incompatible dimensions" +
                                str(self.dims) + ", " + str(other.dims))
            for element in (<QobjEvo> other).elements:
                self.elements.append(element)
            self._update_feedback(other)

        elif isinstance(other, Qobj):
            if other._dims != self._dims:
                raise TypeError("incompatible dimensions" +
                                str(self.dims) + ", " + str(other.dims))
            self.elements.append(_ConstantElement(other))
        elif (
            isinstance(other, numbers.Number) and
            self._dims[0] == self._dims[1]
        ):
            self.elements.append(_ConstantElement(other * qutip.qeye_like(self)))
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
            if left._dims[1] != (<QobjEvo> right)._dims[0]:
                raise TypeError("incompatible dimensions" +
                                 str(left.dims[1]) + ", " +
                                 str((<QobjEvo> right).dims[0]))
            res = right.copy()
            res._dims = Dimensions(left._dims[0], right._dims[1])
            res.shape = (left.shape[0], right.shape[1])
            left = _ConstantElement(left)
            res.elements = [left @ element for element in res.elements]
            res._update_feedback()

            return res

        else:
            return NotImplemented

    def __rmatmul__(QobjEvo self, other):
        cdef QobjEvo res
        if isinstance(other, Qobj):
            if other._dims[1] != self._dims[0]:
                raise TypeError("incompatible dimensions" +
                                 str(other.dims[1]) + ", " +
                                 str(self.dims[0]))
            res = self.copy()
            res._dims = Dimensions([other._dims[0], res._dims[1]])
            res.shape = (other.shape[0], res.shape[1])
            other = _ConstantElement(other)
            res.elements = [other @ element for element in res.elements]
            res._update_feedback()
            return res
        else:
            return NotImplemented

    def __imatmul__(QobjEvo self, other):
        if isinstance(other, (Qobj, QobjEvo)):
            if self._dims[1] != other._dims[0]:
                raise TypeError("incompatible dimensions" +
                                str(self.dims[1]) + ", " +
                                str(other.dims[0]))
            self._dims = Dimensions([self._dims[0], other._dims[1]])
            self.shape = (self.shape[0], other.shape[1])
            if isinstance(other, Qobj):
                other = _ConstantElement(other)
                self.elements = [element @ other for element in self.elements]
                self._update_feedback()

            elif isinstance(other, QobjEvo):
                self.elements = [left @ right
                    for left, right in itertools.product(
                        self.elements, (<QobjEvo> other).elements
                    )]
                self._update_feedback(other)

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
        res._dims = Dimensions(res._dims[0], res._dims[1])
        res.shape = res._dims.shape
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
        res._dims = Dimensions(res._dims[0], res._dims[1])
        res.shape = res._dims.shape
        return res

    def to(self, data_type):
        """
        Convert the underlying data store of all component into the desired
        storage representation.

        The different storage representations available are the "data-layer
        types".  By default, these are :obj:`.Dense`, :obj:`.Dia` and
        :obj:`.CSR`, which respectively construct a dense matrix, diagonal
        sparse matrixand a compressed sparse row one.

        The :obj:`.QobjEvo` is transformed inplace.

        Parameters
        ----------
        data_type : type
            The data-layer type that the data of this :obj:`.Qobj` should be
            converted to.

        Returns
        -------
        None
        """
        return self.linear_map(partial(Qobj.to, data_type=data_type),
                               _skip_check=True)

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

          ``QobjEvo([sigmax(), coeff]).linear_map(spre)``

        gives the same result has

          ``QobjEvo([spre(sigmax()), coeff])``

        Parameters
        ----------
        op_mapping: callable
            Funtion to apply to each elements.

        Returns
        -------
        :class:`.QobjEvo`
            Modified object

        Notes
        -----
        Does not modify the coefficients, thus ``linear_map(conj)`` would not
        give the the conjugate of the QobjEvo. It's only valid for linear
        transformations.
        """
        if not _skip_check:
            out = op_mapping(self(0))
            if not isinstance(out, Qobj):
                raise TypeError("The op_mapping function must return a Qobj")
        cdef QobjEvo res = self.copy()
        res.elements = [element.linear_map(op_mapping) for element in res.elements]
        res._dims = res.elements[0].qobj(0)._dims
        res.shape = res.elements[0].qobj(0).shape
        res._update_feedback()

        if not _skip_check:
            if res(0) != out:
                raise ValueError("The mapping is not linear")

        return res

    ###########################################################################
    # Cleaning and compress                                                   #
    ###########################################################################
    def _compress_merge_qobj(self, coeff_elements):
        """Merge element with matching qobj:
        ``[A, f1], [A, f2] -> [A, f1+f2]``
        """
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
        Look for redundance in the :obj:`.QobjEvo` components:

        Constant parts, (:obj:`.Qobj` without :obj:`Coefficient`) will be
        summed.
        Pairs ``[Qobj, Coefficient]`` with the same :obj:`.Qobj` are merged.

        Example:
        ``[[sigmax(), f1], [sigmax(), f2]] -> [[sigmax(), f1+f2]]``

        The :obj:`.QobjEvo` is transformed inplace.

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

    def to_list(QobjEvo self):
        """
        Restore the QobjEvo to a list form.

        Returns
        -------
        list_qevo: list
            The QobjEvo as a list, element are either :obj:`.Qobj` for
            constant parts, ``[Qobj, Coefficient]`` for coefficient based term.
            The original format of the :obj:`Coefficient` is not restored.
            Lastly if the original :obj:`.QobjEvo` is constructed with a
            function returning a Qobj, the term is returned as a pair of the
            original function and args (``dict``).
        """
        out = []
        for element in self.elements:
            if isinstance(element, _ConstantElement):
                out.append(element.qobj(0))
            elif isinstance(element, _EvoElement):
                coeff = element._coefficient
                out.append([element.qobj(0), coeff])
            elif isinstance(element, _FuncElement):
                func = element._func
                args = element._args
                out.append([func, args])
            else:
                out.append([element, {}])
        return out

    ###########################################################################
    # properties                                                              #
    ###########################################################################
    @property
    def num_elements(self):
        """Number of parts composing the system"""
        return len(self.elements)

    @property
    def isconstant(self):
        """Does the system change depending on ``t``"""
        return not any(type(element) is not _ConstantElement
                       for element in self.elements)

    @property
    def isoper(self):
        """Indicates if the system represents an operator."""
        return self._dims.type in ['oper', 'scalar']

    @property
    def issuper(self):
        """Indicates if the system represents a superoperator."""
        return self._dims.type == 'super'

    @property
    def dims(self):
        return self._dims.as_list()

    @property
    def type(self):
        return self._dims.type

    @property
    def superrep(self):
        return self._dims.superrep

    @property
    def isbra(self):
        """Indicates if the system represents a bra state."""
        return self._dims.type in ['bra', 'scalar']

    @property
    def isket(self):
        """Indicates if the system represents a ket state."""
        return self._dims.type in ['ket', 'scalar']

    @property
    def isoperket(self):
        """Indicates if the system represents a operator-ket state."""
        return self._dims.type == 'operator-ket'

    @property
    def isoperbra(self):
        """Indicates if the system represents a operator-bra state."""
        return self._dims.type == 'operator-bra'

    @property
    def dtype(self):
        """
        Type of the data layers of the QobjEvo.
        When different data layers are used, we return the type of the sum of
        the parts.
        """
        part_types = [part.dtype for part in self.elements]
        if (
            part_types[0] is not None
            and all(part == part_types[0] for part in part_types)
        ):
            return part_types[0]
        return self(0).dtype

    ###########################################################################
    # operation methods                                                       #
    ###########################################################################
    def expect(QobjEvo self, object t, state, check_real=True):
        """
        Expectation value of this operator at time ``t`` with the state.

        Parameters
        ----------
        t : float
            Time of the operator to apply.

        state : Qobj
            right matrix of the product

        check_real : bool (True)
            Whether to convert the result to a `real` when the imaginary part
            is smaller than the real part by a dactor of
            ``settings.core['rtol']``.

        Returns
        -------
        expect : float or complex
            ``state.adjoint() @ self @ state`` if ``state`` is a ket.
            ``trace(self @ matrix)`` is ``state`` is an operator or
            operator-ket.
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
            (self._dims[1] == state._dims[0]) or
            (self.issuper and self._dims[1] == state._dims)
        ):
            raise ValueError("incompatible dimensions " + str(self.dims) +
                             ", " + str(state.dims))
        out = self.expect_data(t, state.data)
        if (
            check_real and
            (out == 0 or (out.real and fabs(out.imag / out.real) < herm_rtol))
        ):
            return out.real
        return out

    cpdef object expect_data(QobjEvo self, object t, Data state):
        """
        Expectation is defined as ``state.adjoint() @ self @ state`` if
        ``state`` is a vector, or ``state`` is an operator and ``self`` is a
        superoperator.  If ``state`` is an operator and ``self`` is an
        operator, then expectation is ``trace(self @ matrix)``.
        """
        if type(state) is Dense:
            return self._expect_dense(t, state)
        cdef _BaseElement part
        cdef object out = 0.
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
            part_data = part.data(t)
            out += part.coeff(t) * expect_func(part_data, state)
        return out

    cdef double complex _expect_dense(QobjEvo self, double t, Dense state) except *:
        """For Dense state, ``column_stack_dense`` can be done inplace if in
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

    def matmul(self, t, state):
        """
        Product of this operator at time ``t`` to the state.
        ``self(t) @ state``

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

        if self._dims[1] != state._dims[0]:
            raise ValueError("incompatible dimensions " + str(self.dims[1]) +
                             ", " + str(state.dims[0]))

        return Qobj(self.matmul_data(t, state.data),
                    dims=[self._dims[0], state._dims[1]],
                    copy=False
                    )

    cpdef Data matmul_data(QobjEvo self, object t, Data state, Data out=None):
        """Compute ``out += self(t) @ state``"""
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


class _Feedback:
    default = None

    def __init__(self):
        raise NotImplementedError("Use subclass")

    def check_consistency(self, dims):
        """
        Raise an error when the dims of the e_ops / state don't match.
        Tell the dims to the feedback for reconstructing the Qobj.
        """
