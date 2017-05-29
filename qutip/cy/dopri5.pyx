# distutils: language = c++

import numpy as np
cimport numpy as np

cdef extern from "src/dopri5.cpp":
    cdef cppclass ode:
        ode(int*, double*, int*, int*, complex*)
        void run(double, double, complex*, complex*)
        int step(double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)
        
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
        return self.cobj.integrate(_t_in,_t_target,rand,<complex*>_psi.data, <double*>_err.data)
        
