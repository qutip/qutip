#cython: language_level=3
cdef extern from "<complex>" namespace "std" nogil:
    double complex cos(double complex x)
    double complex conj(double complex x)
    double         norm(double complex x)
    double complex exp(double complex x)

from qutip.cy.inter import _prep_cubic_spline
from qutip.cy.inter cimport (_spline_complex_cte_second,
                             _spline_complex_t_second,
                             _step_complex_t, _step_complex_cte)
from qutip.cy.interpolate cimport (interp, zinterp)
import numpy as np
cimport numpy as cnp

cdef class Coefficient:
    def __cinit__(self):
        self.codeString = ""

    cdef complex _call(self, double t, dict args):
        # Both def and cdef instead of cpdef to catch mistakes calling the
        # python function when not needed
        return 0j

    def __call__(self, double t, dict args):
        return self._call(t, args)

    def optstr(self):
        return self.codeString

    def __add__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        return Add(self, other)

    def __mul__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        return Mul(self, other)


cdef class FunctionCoefficient(Coefficient):
    cdef object func

    def __init__(self, func):
        self.func = func

    cdef complex _call(self, double t, dict args):
        return self.func(t, args)


cdef class InterpolateCoefficient(Coefficient):
    cdef double a, b
    cdef complex[::1] c

    def __init__(self, splineObj):
        self.a = splineObj.a
        self.b = splineObj.b
        self.c = splineObj.coeffs.astype(np.complex128)

    cdef complex _call(self, double t, dict args):
        return zinterp(t, self.a, self.b, self.c)


cdef class InterCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr, second_derr

    def __init__(self, cnp.ndarray[complex, ndim=1] coeff_arr,
                 cnp.ndarray[double, ndim=1] tlist):
        self.second_derr, self.cte = _prep_cubic_spline(coeff_arr, tlist)
        self.tlist = tlist
        self.coeff_arr = coeff_arr
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    cdef complex _call(self, double t, dict args):
        cdef complex coeff
        if self.cte:
            coeff = _spline_complex_cte_second(t, self.tlist,
                                               self.coeff_arr,
                                               self.second_derr,
                                               self.n_t, self.dt)
        else:
            coeff = _spline_complex_t_second(t, self.tlist,
                                             self.coeff_arr,
                                             self.second_derr,
                                             self.n_t)
        return coeff


cdef class StepCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr

    def __init__(self, complex[::1] coeff_arr, double[::1] tlist):
        self.cte = np.allclose(np.diff(tlist), tlist[1]-tlist[0])
        self.tlist = tlist
        self.coeff_arr = coeff_arr.copy()
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    cdef complex _call(self, double t, dict args):
        cdef complex coeff
        if self.cte:
            coeff = _step_complex_cte(t, self.tlist, self.coeff_arr, self.n_t)
        else:
            coeff = _step_complex_t(t, self.tlist, self.coeff_arr, self.n_t)
        return coeff


cdef class Add(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cdef complex _call(self, double t, dict args):
        return self.first._call(t, args) + self.second._call(t, args)

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})+({})".format(str1, str2)
        return ""


cdef class Mul(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cdef complex _call(self, double t, dict args):
        return self.first._call(t, args) * self.second._call(t, args)

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})*({})".format(str1, str2)
        return ""


cdef class Conj(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t, dict args):
        return conj(self.base._call(t, args))

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "conj({})".format(str1)
        return ""


cdef class Norm(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t, dict args):
        return norm(self.base._call(t, args))

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "norm({})".format(str1)
        return ""


cdef class Shift(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t, dict args):
        cdef _t0 = args["_t0"]
        return self.base._call(t+_t0, args)

    def optstr(self):
        from re import sub
        str1 = self.base.optstr()
        if str1:
            return sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                       "(t+_t0)", " " + str1 + " ")
        return ""
