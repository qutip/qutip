from qutip.core import data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data cimport Data


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data euler(StochasticSystem system, t, Data state, dt, dW):
    """
    Integration scheme:
    Basic Euler order 0.5
    dV = d1 dt + d2_i dW_i
    Numerical Solution of Stochastic Differential Equations
    By Peter E. Kloeden, Eckhard Platen
    """
    cdef int i
    a = system.drift(t, state)
    b = system.diffusion(t, state)
    new_state = _data.add(state, a, dt)
    for i in range(system.num_collapse):
        new_state = _data.add(new_state, b[i], dW[i])
    return new_state


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data platen(StochasticSystem system, t, Data state, dt, dW):
    """
    Platen rhs function for both master eq and schrodinger eq.
    dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
         + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
         + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

    Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
    V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
    The Theory of Open Quantum Systems
    Chapter 7 Eq. (7.47), H.-P Breuer, F. Petruccione
    """
    cdef int i, j
    cdef double sqrt_dt = np.sqrt(dt)
    cdef double sqrt_dt_inv = 0.25/sqrt_dt
    cdef double dw, dw2

    d1 = _data.add(system.drift(t, state), state)
    d2 = system.diffusion(t, state)

    out = _data.mul(d1, 0.5)
    Vt = d1.copy()
    Vp = []
    Vm = []
    for i in range(system.num_collapse):
        Vp.append(_data.add(d1, d2[i], sqrt_dt))
        Vm.append(_data.add(d1, d2[i], -sqrt_dt))
        Vt = _data.add(Vt, d2[i], dW[i])

    d1 = system.drift(t, Vt)
    out = _data.add(out, d1, 0.5)
    out = _data.add(out, state, 0.5)
    for i in range(system.num_collapse):
        d2p = system.diffusion(t, Vp[i])
        d2m = system.diffusion(t, Vm[i])
        dw = dW[i] * 0.25
        out = _data.add(out, d2m[i], dw)
        out = _data.add(out, d2[i], 2 * dw)
        out = _data.add(out, d2p[i], dw)

        for j in range(system.num_collapse):
            dw2 = sqrt_dt_inv * (dW[i] * dW[j] - dt * (i == j))
            out = _data.add(out, d2p[j], dw2)
            out = _data.add(out, d2m[j], -dw2)

    return out
