#cython: language_level=3

cpdef double interp(double x, double a, double b, double[::1] c)

cpdef complex zinterp(double x, double a, double b, complex[::1] c)
