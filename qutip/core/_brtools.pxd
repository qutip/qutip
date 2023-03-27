#cython: language_level=3
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data cimport Data

cdef class SpectraCoefficient(Coefficient):
    cdef Coefficient coeff_t
    cdef Coefficient coeff_w
    cdef double w

cpdef Data matmul_var_data(Data left, Data right, int transleft, int transright)

cdef class _EigenBasisTransform:
    cdef:
        QobjEvo oper
        int size
        readonly bint isconstant
        double _t
        object _eigvals  # np.ndarray
        Data _evecs, _evecs_inv

    cpdef object eigenvalues(self, double t)
    cpdef Data evecs(self, double t)

    cpdef Data to_eigbasis(self, double t, Data fock)
    cpdef Data from_eigbasis(self, double t, Data eig)

    cdef Data _inv(self, double t)
    cdef void _compute_eigen(self, double t) except *

    cdef Data _S_converter(self, double t)
    cdef Data _S_converter_inverse(self, double t)
