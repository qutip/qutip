cdef extern from "complex.h":
    double         cabs(double complex x)
    double complex cacos(double complex x)
    double complex cacosh(double complex x)
    double         carg(double complex x)
    double complex casin(double complex x)
    double complex casinh(double complex x)
    double complex catan(double complex x)
    double complex catanh(double complex x)
    double complex ccos(double complex x)
    double complex ccosh(double complex x)
    double complex cexp(double complex x)
    double         cimag(double complex x)
    double complex clog(double complex x)
    double complex conj(double complex x)
    double complex cpow(double complex x, double complex y)
    double complex cproj(double complex x)
    double         creal(double complex x)
    double complex csin(double complex x)
    double complex csinh(double complex x)
    double complex csqrt(double complex x)
    double complex ctan(double complex x)
    double complex ctanh(double complex x)


cdef double abs(double complex x):
    return cabs(x)
cdef double complex acos(double complex x):
    return cacos(x)
cdef double complex acosh(double complex x):
    return cacosh(x)
cdef double arg(double complex x):
    return carg(x)
cdef double complex asin(double complex x):
    return casin(x)
cdef double complex asinh(double complex x):
    return casinh(x)
cdef double complex atan(double complex x):
    return catan(x)
cdef double complex atanh(double complex x):
    return catanh(x)
cdef double complex cos(double complex x):
    return ccos(x)
cdef double complex cosh(double complex x):
    return ccosh(x)
cdef double complex exp(double complex x):
    return cexp(x)
cdef double imag(double complex x):
    return cimag(x)
cdef double complex log(double complex x):
    return clog(x)
cdef double complex pow(double complex x, double complex y):
    return cpow(x,y)
cdef double complex proj(double complex x):
    return cproj(x)
cdef double real(double complex x):
    return creal(x)
cdef double complex sin(double complex x):
    return csin(x)
cdef double complex sinh(double complex x):
    return csinh(x)
cdef double complex sqrt(double complex x):
    return csqrt(x)
cdef double complex tan(double complex x):
    return ctan(x)
cdef double complex tanh(double complex x):
    return ctanh(x)