#cython: language_level=3

cdef complex _spline_complex_t_second(double x, double[::1] t,
                                      complex[::1] y, complex[::1] M,
                                      int N)

cdef complex _spline_complex_cte_second(double x, double[::1] t,
                                        complex[::1] y, complex[::1] M,
                                        int N, double dt)

cdef double _spline_float_t_second(double x, double[::1] t,
                                   double[::1] y, double[::1] M,
                                   int N)

cdef double _spline_float_cte_second(double x, double[::1] t,
                                     double[::1] y, double[::1] M,
                                     int N, double dt)

cdef double _step_float_cte(double x, double[::1] t, double[::1] y, int n_t)

cdef complex _step_complex_cte(double x, double[::1] t, complex[::1] y, int n_t)

cdef double _step_float_t(double x, double[::1] t, double[::1] y, int n_t)

cdef complex _step_complex_t(double x, double[::1] t, complex[::1] y, int n_t)
