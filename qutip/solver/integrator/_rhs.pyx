#cython: language_level=3

from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data.constant import zeros_like
from qutip.core.data.add import iadd
from collections.abc import Callable

# Migrating integrator from supporting QobjEvo (matmul) only, to functions
# with signature:
#
#     (double t, Data state) -> Data dstate or
#     (double t, Data state, Data dstate)
#
# - Keep fast cython access to `QobjEvo.matmul_data`
# - Ensure support for both inplace and not python function.
#


__all__ = ["RHS"]


cdef class RHS:
    """
    Container for the derivative function in qutip's cython integrator.

    When it's a python callable, wrap it and add inplace support.

    When it's a bound method of QobjEvo: bind the ``QobjEvo.matmul_data``
    cython call directly as the derivative.
    **This ignore the exact method used.**

    Parameters
    ----------
    derivative: Callable[[float, Data, ...], Data] | QobjEvo
        Function to integrate.
        Can be either a QobjEvo where QobjEvo @ state is the derivative or
        a callable. This function can be both inplace or not.

    inplace: bool
        If ``derivative`` is a callable, whether that function take the output
        inplace or not.

    """
    def __init__(
        self,
        derivative: Callable[[float, Data], Data] | Callable[[float, Data, Data], Data] | QobjEvo,
        inplace: bool=False
    ):
        self.derivative = derivative
        self.inplace = inplace
        self.qevo_derr = False
        if (
            isinstance(getattr(derivative, "__self__", None), QobjEvo)
            and getattr(derivative.__self__, "matmul_data", None) == derivative
        ):
            self.qevo = derivative.__self__
            self.qevo_derr = True

    def __call__(self, t: float, state: Data):
        """
        Apply the derivative function
        """
        return self.apply(t, state)

    cdef Data apply(self, double t, Data state, Data out=None):
        """
        Cython interface for the derivative function.
        """
        if self.qevo_derr:
            return self.qevo.matmul_data(t, state, out=out)

        if self.inplace:
            if out is None:
                out = zeros_like(state)
            out = self.derivative(t, state, out)
        elif out is not None:
            out = iadd(self.derivative(t, state), out)
        else:
            out = self.derivative(t, state)
        return out
