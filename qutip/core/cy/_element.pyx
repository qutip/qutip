#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

from .. import data as _data
from qutip.core.data cimport Dense, Data, dense
from qutip.core.data.matmul cimport *
from math import nan as Nan
cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

__all__ = ['_CteElement', '_EvoElement',
           '_FuncElement', '_MapElement', '_ProdElement']


cdef class _BaseElement:
    """
    A representation of a single time-dependent list element in the
    time-dependent operator format used in QobjEvo and solvers.
    For example, in a :obj:`QobjEvo` created by::

      QobjEvo([sigmax(), [sigmay(), 'cos(pi * t)']])

    there will be one `_BaseElement` instance for the `sigmax()` term,
    and one for the `[sigmay(), 'cos(pi*t)']`.

    Time-dependent defined as a function is also represented as a
    `_BaseElement` instance.

    There are 3 methods to obtain the element contrubution: `coeff`, `qobj`
    and `data`. They are all called with a time, returning the respective
    object. The reason there are three separate methods is for speed since
    cython cannot return multiple object without the creation of a python
    object which can be as long as the call itself.

    All `_BaseElement` are immutable and method modifying the content return a
    new instance.
    """
    cpdef Data data(self, double t):
        """
        Return the coefficient at `t`.
        """
        return self._data

    cpdef object qobj(self, double t):
        """
        Return the Qobj at `t`.
        """
        return self._qobj

    cpdef double complex coeff(self, double t) except *:
        """
        Return the Data at `t`.
        """
        return self._coeff

    cdef Data matmul(_BaseElement self, double t, Data state, Data out):
        """
        out += Qobj(t) @ state * coeff(t)
        """
        # matmul is here instead of in QobjEvo.matmul_data to support
        # function element with fast matmul method.
        # TODO: fix to imatmul_data when c-dispatch with inplace support is
        # available
        if type(state) is Dense and type(out) is Dense:
            imatmul_data_dense(self.data(t), state, self.coeff(t), out)
            return out
        else:
            return _data.add(
                out,
                _data.matmul(self.data(t), state, self.coeff(t))
            )

    def replace_arguments(self, new_args, fElem_safe=None):
        """
        Make a copy with new args.
        """
        raise NotImplementedError

    def linear_map(self, function, conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        raise NotImplementedError


cdef class _CteElement(_BaseElement):
    """
    Constant part of a list format :obj:`QobjEvo`.
    A constant :obj:`QobjEvo` will contain one `_CteElement`::

      qevo = QobjEvo(H0)
      qevo.elements = [_CteElement(H0)]
    """
    def __init__(self, qobj):
        self._data = qobj.data
        self._qobj = qobj
        self._coeff = 1.+0j

    def __mul__(left, right):
        cdef _CteElement base
        cdef object factor
        if type(left) is _CteElement:
            base = left
            factor = right
        elif type(right) is _CteElement:
            base = right
            factor = left
        return _CteElement(base._qobj * factor)

    def __matmul__(left, right):
        if (type(left) is _CteElement and type(right) is _CteElement):
            return _CteElement((<_CteElement> left)._qobj *
                               (<_CteElement> right)._qobj)
        return NotImplemented

    def linear_map(_CteElement self, function, conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        return _CteElement(function(self._qobj))

    def replace_arguments(self, new_args, fElem_safe=None):
        """
        Make a copy with new args.
        """
        return self


cdef class _EvoElement(_BaseElement):
    """
    A pair of a :obj:`Qobj` and a :obj:`Coefficient` from the list format
    time-dependent operator::

      qevo = QobjEvo([[H0, coeff0], [H1, coeff1]])
      qevo.elements = [_EvoElement(H0, coeff0), _EvoElement(H1, coeff1)]
    """
    def __init__(self, qobj, coefficient):
        self._qobj = qobj
        self._data = qobj.data
        self.coefficient = coefficient

    cpdef double complex coeff(self, double t) except *:
        """
        Return the Coefficient at `t`.
        """
        return self.coefficient(t)

    def __mul__(left, right):
        cdef _EvoElement base
        cdef object factor
        if type(left) is _EvoElement:
            base = left
            factor = right
        if type(right) is _EvoElement:
            base = right
            factor = left
        return _EvoElement(base._qobj * factor, base.coefficient)

    def __matmul__(left, right):
        if isinstance(left, _EvoElement) and isinstance(right, _EvoElement):
            coefficient = left.coefficient * right.coefficient
        elif isinstance(left, _EvoElement) and isinstance(right, _CteElement):
            coefficient = left.coefficient
        elif isinstance(right, _EvoElement) and isinstance(left, _CteElement):
            coefficient = right.coefficient
        else:
            return NotImplemented
        return _EvoElement(left._qobj * right._qobj, coefficient)

    def linear_map(_EvoElement self, function, conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        return _EvoElement(function(self._qobj),
            self.coefficient.conj() if conjugate else self.coefficient)

    def replace_arguments(self, new_args, fElem_safe=None):
        """
        Make a copy with new args.
        """
        return _EvoElement(
            self._qobj.copy(),
            self.coefficient.replace_arguments(new_args)
        )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# _FuncElement, _MapElement, _ProdElement : support for function based QobjEvo.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cdef class _FuncElement(_BaseElement):
    """
    Used with :obj:`QobjEvo` to build a function with signature: ::

      func(t: float, args: dict) -> Qobj

    :obj:`QobjEvo` created from such a function contian one
    :obj:`_FuncElement`::

      qevo = QobjEvo(func, args=args)
      qevo.elements = [_FuncElement(func, args)]

    This class has basic memoize capacity: it saves the last call ::

        op = QobjEvo(func, args=args)
        (op.dag() * op)(t)

    call the function `func` only once.
    """
    def __init__(self, func, args):
        self.func = func
        self.args = args.copy()
        self._coeff = 1.0
        self._previous = (Nan, None)

    cpdef Data data(self, double t):
        """
        Return the Data at `t`.
        """
        return self.qobj(t).data

    cpdef object qobj(self, double t):
        """
        Return the Qobj at `t`.
        """
        cdef double _t
        cdef object _qobj
        _t, _qobj = self._previous
        if t == _t:
            return _qobj
        _qobj = self.func(t, self.args)
        self._previous = (t, _qobj)
        return _qobj

    def __mul__(left, right):
        cdef _MapElement out
        if type(left) is _FuncElement:
            out = _MapElement(left, [], right)
        if type(right) is _FuncElement:
            out = _MapElement(right, [], left)
        return out

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    def linear_map(_FuncElement self, function, conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        return _MapElement(self, [function])

    def replace_arguments(_FuncElement self, new_args, fElem_safe=None):
        """
        Make a copy with new args.

        A :obj:`QobjEvo` can contain multiple elements with the same instance
        of a :obj:`_FuncElement`. After updating the args, matching instance
        will still be matching to keep the memoize capacity.
        `fElem_safe` is used to ensure this.

        Example::
        With::
        ```
            op = QobjEvo(f, args)
            op2 = op.dag() * op
        ```
        ``op2.elements`` is::
            ``[_ProdElement(_MapElement(f1, [dag]), f1)]``

        with ``f1`` one instance of ``_FuncElement(f, args)``, so after::

            ``new_op = op2.replace_arguments(new_args)``
        ``new_op.elements`` becomes::
            ``_ProdElement(_MapElement(f2, [dag]), f2)``

        Here, ``f2`` is still one instance.
        """
        if fElem_safe is None:
            return _FuncElement(self.func, {**self.args, **new_args})
        for old, new in fElem_safe:
            if old is self:
                return new
        new = _FuncElement(self.func, {**self.args, **new_args})
        fElem_safe.append((self, new))
        return new


cdef class _MapElement(_BaseElement):
    """
    :obj:`_FuncElement` decorated with linear tranformations.

    Linear tranformations available in :obj:`QobjEvo` include transpose,
    adjoint, conjugate, convertion and product with number::
    ```
        op = QobjEvo(f, args=args)
        op2 = op.conj().dag() * 2
    ```
    Then ``op2.elements`` is::
        ``[_MapElement(_FuncElement(f, args), [conj, dag], 2)]``

    """
    def __init__(self, _FuncElement base, transform, coeff=1.):
        self.base = base
        self.transform = transform
        self._coeff = coeff

    cpdef Data data(self, double t):
        """
        Return the Data at `t`.
        """
        return self.qobj(t).data

    cpdef object qobj(self, double t):
        """
        Return the Qobj at `t`.
        """
        out = self.base.qobj(t)
        for func in self.transform:
            out = func(out)
        return out

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
            self.base,
            self.transform.copy(),
            self._coeff*factor
        )

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    def linear_map(_MapElement self, function, conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        return _MapElement(
            self.base,
            self.transform + [function],
            conj(self._coeff) if conjugate else self._coeff
        )

    def replace_arguments(_MapElement self, new_args, fElem_safe=None):
        """
        Make a copy with new args.
        """
        return _MapElement(
            self.base.replace_arguments(new_args, fElem_safe),
            self.transform.copy(),
            self._coeff
        )


cdef class _ProdElement(_BaseElement):
    """
    Product of a :obj:`_FuncElement` or :obj:`_MapElement` with other
    :obj:`_BaseElement`. Include a stack of linear transformation to be
    applied after the product::

        ``op = QobjEvo(f) * qobj1``
    Then ``op.elements`` is::
        ``[_ProdElement(_FuncElement(f, {}), _CteElement(qobj1))]``
    """
    def __init__(self, left, right, transform, conj=False):
        self.left = left
        self.right = right
        self.conj = conj
        self.transform = transform

    cpdef Data data(self, double t):
        """
        Return the Data at `t`.
        """
        return self.qobj(t).data

    cpdef object qobj(self, double t):
        """
        Return the Qobj at `t`.
        """
        out = self.left.qobj(t) @ self.right.qobj(t)
        for func in self.transform:
            out = func(out)
        return out

    cpdef double complex coeff(self, double t) except *:
        """
        Return the Coefficient at `t`.
        """
        cdef double complex out = self.left.coeff(t) * self.right.coeff(t)
        return conj(out) if self.conj else out

    def __mul__(left, right):
        cdef _ProdElement self
        cdef double complex factor
        if type(left) is _ProdElement:
            self = left
            factor = right
        if type(right) is _ProdElement:
            self = right
            factor = left
        return _ProdElement(self.left, self.right * factor,
                            self.transform.copy(), self.conj)

    def __matmul__(left, right):
        return _ProdElement(left, right, [])

    def linear_map(self, function, bool conjugate=False):
        """
        Linear transformation of the Qobj.
        """
        return _ProdElement(
            self.left, self.right,
            self.transform + [function],
            self.conj ^ conjugate
        )

    cdef Data matmul(_ProdElement self, double t, Data state, Data out):
        """
        out += Qobj(t) @ state * coeff(t)
        """
        cdef Data temp
        if not self.transform:
            shape_0 = self.left.qobj(t).shape[0]
            if type(state) is Dense:
                temp = dense.zeros(shape_0, state.shape[1],
                                   (<Dense> state).fortran)
            else:
                temp = _data.zeros[type(state)](shape_0, state.shape[1])
            temp = self.right.matmul(t, state, temp)
            out = self.left.matmul(t, temp, out)
            return out
        elif type(state) is Dense and type(out) is Dense:
            imatmul_data_dense(self.data(t), state, self.coeff(t), out)
            return out
        else:
            return _data.add(
                out,
                _data.matmul(self.data(t), state, self.coeff(t))
            )

    def replace_arguments(_ProdElement self, new_args, fElem_safe=None):
        """
        Make a copy with new args.
        """
        return _ProdElement(
            self.left.replace_arguments(new_args, fElem_safe),
            self.right.replace_arguments(new_args, fElem_safe),
            self.transform.copy(),
            self.conj
        )
