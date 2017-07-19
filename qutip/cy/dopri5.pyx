# distutils: language = c++

import numpy as np
cimport numpy as np

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)


"""cdef extern from "src/dopri5.cpp":
    cdef cppclass ode:
        ode(int*, double*, int*, int*, complex*)
        void run(double, double, complex*, complex*)
        int step(double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)

    #cdef cppclass test:
    #    test()
    #    double run(double, void (*f)(double, double*))
        
cdef class ode_dopri:
    cdef ode* cobj
            
    def __init__(self, int l, _h_ptr, _h_ind, np.ndarray[complex, ndim=1] _h_data, config):
        _y1 = np.zeros(l,dtype=complex)
        cdef np.ndarray[int, ndim=1] h_ptr = _h_ptr.astype(np.intc)
        cdef np.ndarray[int, ndim=1] h_ind = _h_ind.astype(np.intc)
        cdef np.ndarray[int, ndim=1] int_option = np.zeros(2,dtype=np.intc)
        cdef np.ndarray[double, ndim=1] double_option = np.zeros(5,dtype=np.double)
                
        int_option[0]=l
        int_option[1]=config.norm_steps

        double_option[0]=config.options.atol
        double_option[1]=config.options.rtol
        double_option[2]=config.options.min_step
        double_option[3]=config.options.max_step
        double_option[4]=config.norm_tol
        self.cobj = new ode(<int*>int_option.data, <double*>double_option.data, 
                            <int*>h_ptr.data, <int*>h_ind.data, <complex*>_h_data.data)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
            
    def __del__(self):
        del self.cobj
        
    cpdef double integrate(self, double _t_in, double _t_target, double rand, \
            np.ndarray[complex, ndim=1] _psi, np.ndarray[double, ndim=1] _err  ):
        #cdef np.ndarray[complex, ndim=1] y
        #t_out = self.cobj.integrate(t_in,_t_target,rand,<complex*>_psi.data)
        return self.cobj.integrate(_t_in,_t_target,rand,<complex*>_psi.data, <double*>_err.data)"""



cdef extern from "src/dopri5_td.cpp":
    cdef cppclass ode:
        ode(int*, double*, int*, int*, complex*)
        ode(int*, double*, int)
        # void run(double, double, complex*, complex*)
        void set_H(int*, int*, complex*, int*, int*, complex*, void (*)(double, complex*))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)

        
cdef class ode_td_dopri:
    cdef ode* cobj
            
    def __init__(self, int l, _h_ptr, _h_ind, np.ndarray[complex, ndim=1] _h_data, config):
        _y1 = np.zeros(l,dtype=complex)
        cdef np.ndarray[int, ndim=1] h_ptr = _h_ptr.astype(np.intc)
        cdef np.ndarray[int, ndim=1] h_ind = _h_ind.astype(np.intc)
        cdef np.ndarray[int, ndim=1] int_option = np.zeros(2,dtype=np.intc)
        cdef np.ndarray[double, ndim=1] double_option = np.zeros(5,dtype=np.double)
                
        int_option[0]=l
        int_option[1]=config.norm_steps

        double_option[0]=config.options.atol
        double_option[1]=config.options.rtol
        double_option[2]=config.options.min_step
        double_option[3]=config.options.max_step
        double_option[4]=config.norm_tol
        self.cobj = new ode(<int*>int_option.data, <double*>double_option.data, 
                            <int*>h_ptr.data, <int*>h_ind.data, <complex*>_h_data.data)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
        
    def __init__(self, int l, 
                 _h_cte_ptr, _h_cte_ind, np.ndarray[complex, ndim=1] _h_cte_data, 
                 _h_ptr, _h_ind, np.ndarray[complex, ndim=1] _h_data, 
                 coeff_function, config):

        _y1 = np.zeros(l,dtype=complex)
        cdef np.ndarray[int, ndim=1] h_cte_ptr = _h_cte_ptr.astype(np.intc)
        cdef np.ndarray[int, ndim=1] h_cte_ind = _h_cte_ind.astype(np.intc)
        cdef np.ndarray[int, ndim=1] h_ptr = _h_ptr.astype(np.intc)
        cdef np.ndarray[int, ndim=1] h_ind = _h_ind.astype(np.intc)
        cdef np.ndarray[int, ndim=1] int_option = np.zeros(2,dtype=np.intc)
        cdef np.ndarray[double, ndim=1] double_option = np.zeros(5,dtype=np.double)
        cdef void* f_ptr = PyLong_AsVoidPtr(coeff_function)
        cdef void (*f)(double, complex*) 
        f = <void (*)(double, complex*)> f_ptr

        int_option[0]=l
        int_option[1]=config.norm_steps

        double_option[0]=config.options.atol
        double_option[1]=config.options.rtol
        double_option[2]=config.options.min_step
        double_option[3]=config.options.max_step
        double_option[4]=config.norm_tol

        self.cobj = new ode(<int*>int_option.data, <double*>double_option.data, config.H_len)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')
        self.cobj.set_H(<int*>h_cte_ptr.data, <int*>h_cte_ind.data, <complex*>_h_cte_data.data,
                        <int*>h_ptr.data, <int*>h_ind.data, <complex*>_h_data.data, f)


    def __del__(self):
        del self.cobj
        
    cpdef double integrate(self, double _t_in, double _t_target, double rand, \
            np.ndarray[complex, ndim=1] _psi, np.ndarray[double, ndim=1] _err  ):
        #cdef np.ndarray[complex, ndim=1] y
        #t_out = self.cobj.integrate(t_in,_t_target,rand,<complex*>_psi.data)
        return self.cobj.integrate(_t_in,_t_target,rand,<complex*>_psi.data, <double*>_err.data)






#cdef void test_func(double t, double* out):
#    out[0] = t**2 + 1.
#    out[1] = t**2 - 1.

#cpdef get_func():
#    cdef void* f_ptr = <void*> test_func
#    return PyLong_FromVoidPtr(f_ptr)














