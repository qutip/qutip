#cython: language_level=3
# Verner 7 Efficient
# http://people.math.sfu.ca/~jverner/RKV76.IIa.Efficient.00001675585.081206.CoeffsOnlyFLOAT
from qutip.solver.ode.explicit_rk cimport Explicit_RungeKutta
from .wrapper cimport QtOdeData, QtOdeFuncWrapper

cdef class vern7(Explicit_RungeKutta):
    cdef QtOdeData _y_8, _y_9
