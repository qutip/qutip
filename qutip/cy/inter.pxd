
cdef complex interpolate(double t, double* str_array_0, int N, double dt)

cdef complex zinterpolate(double t, complex* str_array_0, int N, double dt)


cpdef complex spline_complex_t_second(double x, double[::1] t,
                                      complex[::1] y, complex[::1] M,
                                      int N)

cpdef complex spline_complex_cte_second(double x, double[::1] t,
                                        complex[::1] y, complex[::1] M,
                                        int N, double dt)

cpdef complex spline_complex_t_poly(double x, double[::1] t,
                                    complex[:,::1] poly,
                                    int N)

cpdef complex spline_complex_cte_poly(double x, double[::1] t,
                                      complex[:,::1] poly,
                                      int N, double dt)

cpdef double spline_float_t_second(double x, double[::1] t,
                                   double[::1] y, double[::1] M,
                                   int N)

cpdef double spline_float_cte_second(double x, double[::1] t,
                                     double[::1] y, double[::1] M,
                                     int N, double dt)

cpdef double spline_float_cte_poly(double x, double[::1] t,
                                   double[:,::1] poly,
                                   int N, double dt)

cpdef double spline_float_t_poly(double x, double[::1] t,
                                 double[:,::1] poly,
                                 int N)
>>>>>>> tdQobj
