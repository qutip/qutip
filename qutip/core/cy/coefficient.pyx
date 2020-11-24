#cython: language_level=3
from .inter import _prep_cubic_spline
from .inter cimport (_spline_complex_cte_second,
                     _spline_complex_t_second,
                     _step_complex_t, _step_complex_cte)
from .interpolate cimport interp, zinterp
import pickle
import scipy
import numpy as np
cimport numpy as cnp
cimport cython
import qutip

cdef extern from "<complex>" namespace "std" nogil:
    double complex cos(double complex x)
    double complex conj(double complex x)
    double         norm(double complex x)
    double complex exp(double complex x)

cdef class Coefficient:
    cdef double complex _call(self, double t) except *:
        return 0j

    cpdef void arguments(self, dict args) except *:
        self.args = args

    def __call__(self, double t, dict args={}):
        """update args and return the
        """
        if args:
            self.arguments(args)
        return self._call(t)

    def optstr(self):
        return ""

    def __add__(left, right):
        if (isinstance(left, InterCoefficient) and isinstance(right, InterCoefficient)):
            return add_inter(left, right)
        if isinstance(left, Coefficient) and isinstance(right, Coefficient):
            return SumCoefficient(left.copy(), right.copy())
        return NotImplemented

    def __mul__(left, right):
        if isinstance(left, Coefficient) and isinstance(right, Coefficient):
            return MulCoefficient(left.copy(), right.copy())
        if isinstance(left, qutip.Qobj):
            return qutip.QobjEvo([left.copy(), right.copy()])
        if isinstance(right, qutip.Qobj):
            return qutip.QobjEvo([right.copy(), left.copy()])
        return NotImplemented

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def conj(self):
        return ConjCoefficient(self.copy())

    def _cdc(self):
        return NormCoefficient(self.copy())

    def _shift(self):
        return ShiftCoefficient(self.copy(), 0)


@cython.auto_pickle(True)
cdef class FunctionCoefficient(Coefficient):
    cdef object func

    def __init__(self, func, args):
        self.func = func
        self.args = args

    cdef complex _call(self, double t) except *:
        return self.func(t, self.args)

    def copy(self):
        return FunctionCoefficient(self.func, self.args.copy())


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


cdef class StrFunctionCoefficient(Coefficient):
    cdef object func
    cdef str base

    str_env = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "pi": np.pi,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "erf": scipy.special.erf,
        "zerf": scipy.special.erf,
        "sqrt": np.sqrt,
        "real": np.real,
        "imag": np.imag,
        "conj": np.conj,
        "abs": np.abs,
        "norm": lambda x: np.abs(x)**2,
        "arg": np.angle,
        "proj": proj,
        "np": np,
        "spe": scipy.special}

    def __init__(self, base, args):
        code = """
def coeff(t, args):
{}
    return {}""".format(
            "\n".join(["    {} = args['{}']".format(key, key) for key in args]),
            base)
        lc = {}
        exec(code, self.str_env, lc)
        self.base = base
        self.func = lc["coeff"]
        self.args = args

    cdef complex _call(self, double t) except *:
        return self.func(t, self.args)

    def copy(self):
        return StrFunctionCoefficient(self.base, self.args.copy())

    def __reduce__(self):
        return (StrFunctionCoefficient, (self.base, self.args))


cdef class InterpolateCoefficient(Coefficient):
    cdef double lower_bound, higher_bound
    cdef complex[::1] spline_data
    cdef object spline

    def __init__(self, splineObj):
        self.lower_bound = splineObj.a
        self.higher_bound = splineObj.b
        self.spline_data = splineObj.coeffs.astype(np.complex128)
        self.spline = splineObj

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return zinterp(t,
                       self.lower_bound,
                       self.higher_bound,
                       self.spline_data)

    def __reduce__(self):
        return InterpolateCoefficient, (self.spline,)


cdef class InterCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr, second_derr
    cdef object tlist_np, coeff_np, second_np

    def __init__(self, coeff_arr, tlist, second=None, cte=None):
        self.tlist_np = tlist
        self.tlist = tlist
        self.coeff_np = coeff_arr
        self.coeff_arr = coeff_arr
        if second is None:
            self.second_np, self.cte = _prep_cubic_spline(coeff_arr, tlist)
        else:
            self.second_np = second
            self.cte = cte
        self.second_derr = self.second_np
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.cte:
            coeff = _spline_complex_cte_second(t,
                                               self.tlist,
                                               self.coeff_arr,
                                               self.second_derr,
                                               self.n_t,
                                               self.dt)
        else:
            coeff = _spline_complex_t_second(t,
                                             self.tlist,
                                             self.coeff_arr,
                                             self.second_derr,
                                             self.n_t)
        return coeff

    def __reduce__(self):
        return (InterCoefficient,
                (self.coeff_np, self.tlist_np, self.second_np, self.cte))

    @property
    def array(self):
        # Fro QIP tests
        return self.coeff_np


cdef Coefficient add_inter(InterCoefficient left, InterCoefficient right):
    if np.array_equal(left.tlist_np, right.tlist_np):
        return InterCoefficient(left.coeff_np + right.coeff_np,
                                left.tlist_np,
                                left.second_np + right.second_np,
                                left.cte
                               )
    else:
        return SumCoefficient(left.copy(), right.copy())


cdef class StepCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr
    cdef object tlist_np, coeff_np

    def __init__(self, coeff_arr, tlist, cte=None):
        self.tlist_np = tlist
        self.tlist = self.tlist_np
        self.coeff_np = coeff_arr
        self.coeff_arr = self.coeff_np
        if cte is None:
            self.cte = np.allclose(np.diff(tlist), tlist[1]-tlist[0])
        else:
            self.cte = cte
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.cte:
            coeff = _step_complex_cte(t, self.tlist, self.coeff_arr, self.n_t)
        else:
            coeff = _step_complex_t(t, self.tlist, self.coeff_arr, self.n_t)
        return coeff

    def __reduce__(self):
        return (StepCoefficient, (self.coeff_np, self.tlist_np, self.cte))

    @property
    def array(self):
        # Fro QIP tests
        return np.array(self.coeff_arr)


@cython.auto_pickle(True)
cdef class SumCoefficient(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cpdef void arguments(self, dict args) except *:
        self.first.arguments(args)
        self.second.arguments(args)

    cdef complex _call(self, double t) except *:
        return self.first._call(t) + self.second._call(t)

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})+({})".format(str1, str2)
        return ""

    def copy(self):
        return SumCoefficient(self.first.copy(), self.second.copy())


@cython.auto_pickle(True)
cdef class MulCoefficient(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cpdef void arguments(self, dict args) except *:
        self.first.arguments(args)
        self.second.arguments(args)

    cdef complex _call(self, double t) except *:
        return self.first._call(t) * self.second._call(t)

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})*({})".format(str1, str2)
        return ""

    def copy(self):
        return MulCoefficient(self.first.copy(), self.second.copy())


@cython.auto_pickle(True)
cdef class ConjCoefficient(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cpdef void arguments(self, dict args) except *:
        self.base.arguments(args)

    cdef complex _call(self, double t) except *:
        return conj(self.base._call(t))

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "conj({})".format(str1)
        return ""

    def copy(self):
        return ConjCoefficient(self.base.copy())


@cython.auto_pickle(True)
cdef class NormCoefficient(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cpdef void arguments(self, dict args) except *:
        self.base.arguments(args)

    cdef complex _call(self, double t) except *:
        return norm(self.base._call(t))

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "norm({})".format(str1)
        return ""

    def copy(self):
        return NormCoefficient(self.base.copy())


@cython.auto_pickle(True)
cdef class ShiftCoefficient(Coefficient):
    cdef Coefficient base
    cdef double _t0

    def __init__(self, Coefficient base, double _t0):
        self.base = base
        self._t0 = _t0

    cpdef void arguments(self, dict args) except *:
        self._t0 = args["_t0"]
        self.base.arguments(args)

    cdef complex _call(self, double t) except *:
        return self.base._call(t + self._t0)

    def optstr(self):
        from re import sub
        str1 = self.base.optstr()
        if str1:
            return sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                       "(t+_t0)", " " + str1 + " ")
        return ""

    def copy(self):
        return ShiftCoefficient(self.base.copy(), self._t0)
