#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdvision=True
#cython: c_api_binop_methods=True

from .. import data as _data
from qutip.core.cy.coefficient import coefficient_function_parameters
from qutip.core.data cimport Dense, Data, dense
from qutip.core.data.matmul cimport *
from math import nan as Nan
cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

__all__ = ['_ConstantElement', '_EvoElement',
           '_FuncElement', '_MapElement', '_ProdElement']


cdef class _BaseElement:
    """
    The representation of a single time-dependent term in the list of
    terms used by QobjEvo and solvers to describe operators.

    Conceptually each term is given by ``coeff(t) * qobj(t)`` where
    ``coeff`` is a complex coefficient and ``qobj`` is a :obj:`.Qobj`. Both
    are functions of time. :meth:`~_BaseElement.coeff` returns the
    coefficient at ``t``. :meth:`~_BaseElement.qobj` returns the :obj:`.Qobj`.

    For example, a :obj:`.QobjEvo` instance created by::

      QobjEvo([sigmax(), [sigmay(), 'cos(pi * t)']])

    would contain one :obj:`~_BaseElement` instance for the ``sigmax()`` term,
    and one for the ``[sigmay(), 'cos(pi*t)']`` term.

    :obj:`~_BaseElement` defines the interface to time-dependent terms.
    Sub-classes implement terms defined in different ways.
    For example, :obj:`~_ConstantElement` implements a term that
    consists only of a constant :obj:`.Qobj` (i.e. where there is no dependence
    on ``t``), :obj:`~_EvoElement` implements a term that consists of a
    time-dependet :obj:`.Coefficient` times a constant :obj:`.Qobj`, and
    so on.

    .. note::

      There are three methods to return the factors of the term.

      :meth:`~_BaseElement.coeff` and :meth:`~_BaseElement.qobj` return the
      coefficient and operator factors respectively. They are separate to
      avoid constructing an intermediate Python object when called from
      Cython code (which may take as long as the rest of the call).

      :meth:`~BaseElement.data` is equivalent to ``.qobj(t).data`` and
      provides a convenience method for this common operation.

    .. note::

      All :obj:`~_BaseElement` instances are immutable and methods that would
      modify an object return a new instance instead.
    """
    cpdef Data data(self, t):
        """
        Returns the underlying :obj:`~Data` of the :obj:`.Qobj` component
        of the term at time ``t``.

        Parameters
        ----------
        t : double
          The time, ``t``.

        Returns
        -------
        :obj:`~Data`
          The underlying data of the :obj:`.Qobj` component of the term
          at time ``t``.
        """
        raise NotImplementedError(
          "Sub-classes of _BaseElement should implement .data(t)."
        )

    cpdef object qobj(self, t):
        """
        Returns the :obj:`.Qobj` component of the term at time ``t``.

        Parameters
        ----------
        t : float
          The time, ``t``.

        Returns
        -------
        :obj:`.Qobj`
          The :obj:`.Qobj` component of the term at time ``t``.
        """
        raise NotImplementedError(
          "Sub-classes of _BaseElement should implement .qobj(t)."
        )

    cpdef object coeff(self, t):
        """
        Returns the complex coefficient of the term at time ``t``.

        Parameters
        ----------
        t : float
          The time, ``t``.

        Returns
        -------
        complex
          The complex coefficient of the term at time ``t``.
        """
        raise NotImplementedError(
          "Sub-classes of _BaseElement should implement .coeff(t)."
        )

    cdef Data matmul_data_t(_BaseElement self, t, Data state, Data out=None):
        """
        Possibly in-place multiplication and addition. Multiplies a given state
        by the elemen's value at time ``t`` and adds the result to ``out``.
        Equivalent to::

          out += self.coeff(t) * self.qobj(t) @ state

        Sub-classes may override :meth:`~matmul_data_t` to provide an more
        efficient implementation.

        Parameters
        ----------
        t : double
          The time, ``t``.

        state : :obj:`~Data`
          The state to multiply by the element term.

        out : :obj:`~Data` or ``NULL``
          The output to add the result of the multiplication to. If ``NULL``,
          the result of the multiplication is returned directly (i.e. ``out``
          is assumed to be the zero matrix).

        Returns
        -------
        data
          The result of ``self.coeff(t) * self.qobj(t) @ state + out``, with
          the addition possibly having been performed in-place on ``out``.

        .. note::

          Because of the possibly but not definitely in-place behaviour of
          this function, the result should always be assigned to some variable
          and surrounding code should assume that ``out`` maybe have been
          modified and that the result may be a reference to ``out``. The
          safest is simply to write ``out = elem.matmul_data_t(t, state, out)``.

        .. note::

          This method would ideally be implemented using a special
          function dispatched by the data layer so that it did not have to
          special case the ``Dense`` operation itself, but the data layer
          dispatch does not yet support in-place operations. Once it does
          this method should be updated to use the new support.
        """
        if out is None:
            return _data.matmul(self.data(t), state, self.coeff(t))
        elif type(state) is Dense and type(out) is Dense:
            imatmul_data_dense(self.data(t), state, self.coeff(t), out)
            return out
        else:
            return _data.add(
                out,
                _data.matmul(self.data(t), state, self.coeff(t))
            )

    def linear_map(self, f, anti=False):
        """
        Return a new element representing a linear transformation ``f``
        of the :obj:`.Qobj` portion of this element and possibly a
        complex conjucation of the coefficient portion (when ``f`` is an
        antilinear map).

        If this element represents ``coeff * qobj`` the returned element
        represents ``coeff * f(obj)`` (if ``anti=False``) or
        ``conj(coeff) * f(obj)`` (if ``anti=True``).

        Parameters
        ----------
        f : function
          The linear transformation to apply to the :obj:`.Qobj` of this
          element.
        anti : bool
          Whether to take the complex conjugate of the coefficient. Default
          is ``False``. Should be set to ``True`` if ``f`` is an antilinear
          map such as the adjoint (i.e. dagger).

        Returns
        -------
        _BaseElement
          A new element with the transformation applied.
        """
        raise NotImplementedError(
          "Sub-classes of _BaseElement should implement .linear_map(t)."
        )

    def replace_arguments(self, args, cache=None):
        """
        Return a copy of the element with the (possible) additional arguments
        to any time-dependent functions updated to the given argument values.
        The arguments of any contained :obj:`.Coefficient` instances are also
        replaced.

        If the operation does not modify this element, the original element
        may be returned.

        Parameters
        ----------
        args : dict
          A dictionary of arguments to update. Keys are the names of the
          arguments and values are the new argument values. Arguments
          not included retain their original values.

        cache : list or ``None``
          A cache to add updated elements to. Unmodified elements are not
          added to the cache. If a cache is supplied and ``.replace_arguments``
          would be called again on the same element with the same arguments,
          the new element from the cache will be returned instead.

          By default the cache is ``None`` and no caching is performed. Cache
          users should supply either ``cache=[]`` (which activates caching)
          or an existing cache (for example, if making calls on sub-elements
          of some composite element).

        Returns
        -------
        _BaseElement
          A new element with the arguments replaced, or possibly this element
          if it would not be modified.

        Example
        -------

        If ``elem`` is a product element for ``op * op.dag()`` it will contain
        two references to the same element for ``op``. Calling::

          elem = elem.replace_arguments({"theta": 3.1416})

        will return a new element that contains two *different* copies of
        ``op.replace_arguments({"theta": 3.1416})``. This will cause ``op``
        to be evaluated *twice* when ``elem.qobj(t)`` is called.

        Calling instead::

          elem = elem.replace_arguments({"theta": 3.1416}, cache=[])

        will return a new element that contains two references to *one* copy
        of ``op.replace_arguments({"theta": 3.1416})``, which will improve
        performance when calling ``elem.qobj(t)`` later.
        """
        raise NotImplementedError(
          "Sub-classes of _BaseElement should implement .replace_arguments(t)."
        )

    def __call__(self, t, args=None):
        if args:
            cache = []
            new = self.replace_arguments(args, cache)
            return new.qobj(t) * new.coeff(t)
        return self.qobj(t) * self.coeff(t)

    @property
    def dtype(self):
        return None


cdef class _ConstantElement(_BaseElement):
    """
    Constant part of a list format :obj:`.QobjEvo`.
    A constant :obj:`.QobjEvo` will contain one `_ConstantElement`::

      qevo = QobjEvo(H0)
      qevo.elements = [_ConstantElement(H0)]
    """
    def __init__(self, qobj):
        self._qobj = qobj
        self._data = self._qobj.data

    def __mul__(left, right):
        if type(left) is _ConstantElement:
            return _ConstantElement((<_ConstantElement> left)._qobj * right)
        elif type(right) is _ConstantElement:
            return _ConstantElement((<_ConstantElement> right)._qobj * left)
        return NotImplemented

    def __matmul__(left, right):
        if type(left) is _ConstantElement and type(right) is _ConstantElement:
            return _ConstantElement(
                (<_ConstantElement> left)._qobj @
                (<_ConstantElement> right)._qobj
            )
        return NotImplemented

    cpdef Data data(self, t):
        return self._data

    cpdef object qobj(self, t):
        return self._qobj

    cpdef object coeff(self, t):
        return 1.

    def linear_map(self, f, anti=False):
        return _ConstantElement(f(self._qobj))

    def replace_arguments(self, args, cache=None):
        return self

    def __call__(self, t, args=None):
        return self._qobj

    @property
    def dtype(self):
        return type(self._data)


cdef class _EvoElement(_BaseElement):
    """
    A pair of a :obj:`.Qobj` and a :obj:`.Coefficient` from the list format
    time-dependent operator::

      qevo = QobjEvo([[H0, coeff0], [H1, coeff1]])
      qevo.elements = [_EvoElement(H0, coeff0), _EvoElement(H1, coeff1)]
    """
    def __init__(self, qobj, coefficient):
        self._qobj = qobj
        self._data = self._qobj.data
        self._coefficient = coefficient

    def __mul__(left, right):
        cdef _EvoElement base
        cdef object factor
        if type(left) is _EvoElement:
            base = left
            factor = right
        if type(right) is _EvoElement:
            base = right
            factor = left
        return _EvoElement(base._qobj * factor, base._coefficient)

    def __matmul__(left, right):
        if isinstance(left, _EvoElement) and isinstance(right, _EvoElement):
            coefficient = left._coefficient * right._coefficient
        elif isinstance(left, _EvoElement) and isinstance(right, _ConstantElement):
            coefficient = left._coefficient
        elif isinstance(right, _EvoElement) and isinstance(left, _ConstantElement):
            coefficient = right._coefficient
        else:
            return NotImplemented
        return _EvoElement(left._qobj * right._qobj, coefficient)

    cpdef Data data(self, t):
        return self._data

    cpdef object qobj(self, t):
        return self._qobj

    cpdef object coeff(self, t):
        return self._coefficient(t)

    def linear_map(self, f, anti=False):
        return _EvoElement(
            f(self._qobj),
            self._coefficient.conj() if anti else self._coefficient,
        )

    def replace_arguments(self, args, cache=None):
        return _EvoElement(
            self._qobj,
            self._coefficient.replace_arguments(args)
        )

    @property
    def dtype(self):
        return type(self._data)


cdef class _FuncElement(_BaseElement):
    """
    Used with :obj:`.QobjEvo` to build an evolution term from a function with
    either the signature: ::

        func(t: float, ...) -> Qobj

    or the older QuTiP 4 style signature: ::

        func(t: float, args: dict) -> Qobj

    In the new style, ``func`` may accept arbitrary named arguments and
    is called as ``func(t, **args)``.

    A :obj:`.QobjEvo` created from such a function contains one
    :obj:`_FuncElement`: ::

        qevo = QobjEvo(func, args=args)
        qevo.elements = [_FuncElement(func, args)]

    Each :obj:`_FuncElement` consists of an immutable pair ``(func, args)``
    of function and argument.

    This class has a basic capacity to memoize calls to ``func`` by saving the
    result of the last call ::

        op = QobjEvo(func, args=args)
        (op.dag() * op)(t)

    calls ``func`` only once.

    Parameters
    ----------
    func : callable(t : float, ...) -> Qobj
        Function returning the element value at time ``t``.

    args : dict
        Values of the arguments to pass to ``func``.

    style : {None, "pythonic", "dict", "auto"}
        The style of the signature used. If style is ``None``,
        the value of ``qutip.settings.core["function_coefficient_style"]``
        is used. Otherwise the supplied value overrides the global setting.

    The parameters ``_f_pythonic`` and ``_f_parameters`` override function
    style and parameter detection and are not intended to be part of
    the public interface.
    """

    _UNSET = object()

    def __init__(self, func, args, style=None, _f_pythonic=_UNSET, _f_parameters=_UNSET):
        if _f_pythonic is self._UNSET or _f_parameters is self._UNSET:
            if not (_f_pythonic is self._UNSET and _f_parameters is self._UNSET):
                raise TypeError(
                    "_f_pythonic and _f_parameters should always be given together."
                )
            _f_pythonic, _f_parameters = coefficient_function_parameters(
                func, style=style)
            if _f_parameters is not None:
                args = {k: args[k] for k in _f_parameters & args.keys()}
            else:
                args = args.copy()
        self._func = func
        self._args = args
        self._f_pythonic = _f_pythonic
        self._f_parameters = _f_parameters
        self._previous = (Nan, None)

    def __mul__(left, right):
        cdef _MapElement out
        if type(left) is _FuncElement:
            out = _MapElement(left, [], right)
        if type(right) is _FuncElement:
            out = _MapElement(right, [], left)
        return out

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    cpdef Data data(self, t):
        return self.qobj(t).data

    cpdef object qobj(self, t):
        cdef double _t
        cdef object _qobj
        _t, _qobj = self._previous
        if t == _t:
            return _qobj
        if self._f_pythonic:
            _qobj = self._func(t, **self._args)
        else:
            _qobj = self._func(t, self._args)
        self._previous = (t, _qobj)
        return _qobj

    cpdef object coeff(self, t):
        return 1.

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(_FuncElement self, args, cache=None):
        if self._f_parameters is not None:
            args = {k: args[k] for k in self._f_parameters & args.keys()}
        if not args:
            return self
        if cache is not None:
            for old, new in cache:
                if old is self:
                    return new
        new = _FuncElement(
                self._func,
                {**self._args, **args},
                _f_pythonic=self._f_pythonic,
                _f_parameters=self._f_parameters,
        )
        if cache is not None:
            cache.append((self, new))
        return new


cdef class _MapElement(_BaseElement):
    """
    :obj:`_FuncElement` decorated with linear tranformations.

    Linear tranformations available in :obj:`.QobjEvo` include transpose,
    adjoint, conjugate, convertion and product with number::
    ```
        op = QobjEvo(f, args=args)
        op2 = op.conj().dag() * 2
    ```
    Then ``op2.elements`` is::
        ``[_MapElement(_FuncElement(f, args), [conj, dag], 2)]``

    """
    def __init__(self, _FuncElement base, transform, coeff=1.):
        self._base = base
        self._transform = transform
        self._coeff = coeff

    def __mul__(left, right):
        cdef _MapElement out, self
        cdef double complex factor
        if type(left) is _MapElement:
            self = left
            factor = right
        elif type(right) is _MapElement:
            self = right
            factor = left
        return _MapElement(
            self._base,
            self._transform.copy(),
            self._coeff*factor
        )

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    cpdef Data data(self, t):
        return self.qobj(t).data

    cpdef object qobj(self, t):
        out = self._base.qobj(t)
        for func in self._transform:
            out = func(out)
        return out

    cpdef object coeff(self, t):
        return self._coeff

    def linear_map(self, f, anti=False):
        return _MapElement(
            self._base,
            self._transform + [f],
            conj(self._coeff) if anti else self._coeff
        )

    def replace_arguments(_MapElement self, args, cache=None):
        return _MapElement(
            self._base.replace_arguments(args, cache=cache),
            self._transform.copy(),
            self._coeff,
        )


cdef class _ProdElement(_BaseElement):
    """
    Product of a :obj:`_FuncElement` or :obj:`_MapElement` with other
    :obj:`_BaseElement`. Include a stack of linear transformation to be
    applied after the product::

        ``op = QobjEvo(f) * qobj1``
    Then ``op.elements`` is::
        ``[_ProdElement(_FuncElement(f, {}), _ConstantElement(qobj1))]``
    """
    def __init__(self, left, right, transform, conj=False):
        self._left = left
        self._right = right
        self._conj = conj
        self._transform = transform

    def __mul__(left, right):
        cdef _ProdElement self
        cdef double complex factor
        if type(left) is _ProdElement:
            self = left
            factor = right
        if type(right) is _ProdElement:
            self = right
            factor = left
        return _ProdElement(self._left, self._right * factor,
                            self._transform.copy(), self._conj)

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    cpdef Data data(self, t):
        return self.qobj(t).data

    cpdef object qobj(self, t):
        out = self._left.qobj(t) @ self._right.qobj(t)
        for func in self._transform:
            out = func(out)
        return out

    cpdef object coeff(self, t):
        cdef double complex out = self._left.coeff(t) * self._right.coeff(t)
        return conj(out) if self._conj else out

    cdef Data matmul_data_t(_ProdElement self, t, Data state, Data out=None):
        cdef Data temp
        if not self._transform:
            temp = self._right.matmul_data_t(t, state)
            out = self._left.matmul_data_t(t, temp, out)
            return out
        elif type(state) is Dense and type(out) is Dense:
            imatmul_data_dense(self.data(t), state, self.coeff(t), out)
            return out
        else:
            return _data.add(
                out,
                _data.matmul(self.data(t), state, self.coeff(t))
            )

    def linear_map(self, f, anti=False):
        return _ProdElement(
            self._left, self._right,
            self._transform + [f],
            self._conj ^ anti
        )

    def replace_arguments(_ProdElement self, args, cache=None):
        return _ProdElement(
            self._left.replace_arguments(args, cache=cache),
            self._right.replace_arguments(args, cache=cache),
            self._transform.copy(),
            self._conj
        )
