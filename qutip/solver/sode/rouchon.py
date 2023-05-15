import numpy as np
from qutip import unstack_columns, stack_columns
from qutip.core import data as _data
from ..stochastic import StochasticSolver
from .sode import SIntegrator
from ..integrator.integrator import Integrator


__all__ = ["RouchonSODE"]


class RouchonSODE(SIntegrator):
    """
    Stochastic integration method keeping the positivity of the density matrix.
    See eq. (4) Pierre Rouchon and Jason F. Ralpha,
    *Efficient Quantum Filtering for Quantum Feedback Control*,
    `arXiv:1410.5345 [quant-ph] <https://arxiv.org/abs/1410.5345>`_,
    Phys. Rev. A 91, 012118, (2015).

    - Order: strong 1

    .. note::

        This method should be used with very small ``dt``. Unlike other
        methods that will return unphysical state (negative eigenvalues, Nans)
        when the time step is too large, this method will return state that
        seems normal.
    """
    integrator_options = {
        "dt": 0.0001,
        "tol": 1e-7,
    }

    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options

        self.H = rhs.H
        if self.H.issuper:
            raise TypeError("The rouchon stochastic integration method can't"
                            " use a premade Liouvillian.")
        self._issuper = rhs.issuper

        dtype = type(self.H(0).data)
        self.c_ops = rhs.c_ops
        self.sc_ops = rhs.sc_ops
        self.cpcds = [op + op.dag() for op in self.sc_ops]
        for op in self.cpcds:
            op.compress()
        self.M = (
            - 1j * self.H
            - sum(op.dag() @ op for op in self.c_ops) * 0.5
            - sum(op.dag() @ op for op in self.sc_ops) * 0.5
        )
        self.M.compress()

        self.num_collapses = len(self.sc_ops)
        self.scc = [
            [self.sc_ops[i] @ self.sc_ops[j] for i in range(j+1)]
            for j in range(self.num_collapses)
        ]

        self.id = _data.identity[dtype](self.H.shape[0])

    def integrate(self, t, copy=True):
        delta_t = (t - self.t)
        if delta_t < 0:
            raise ValueError("Stochastic integration need increasing times")
        elif delta_t == 0:
            return self.t, self.state, np.zeros(self.N_dw)

        dt = self.options["dt"]
        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > self.options["tol"]:
            # Not a whole number of steps.
            N += 1
            dt = delta_t / N
        dW = self.generator.normal(
            0,
            np.sqrt(dt),
            size=(N, self.num_collapses)
        )

        if self._issuper:
            self.state = unstack_columns(self.state)
        for dw in dW:
            self.state = self._step(self.t, self.state, dt, dw)
            self.t += dt
        if self._issuper:
            self.state = stack_columns(self.state)

        return self.t, self.state, np.sum(dW, axis=0)

    def _step(self, t, state, dt, dW):
        dy = [
            op.expect_data(t, state) * dt + dw
            for op, dw in zip(self.cpcds, dW)
        ]
        M = _data.add(self.id, self.M._call(t), dt)
        for i in range(self.num_collapses):
            M = _data.add(M, self.sc_ops[i]._call(t), dy[i])
            M = _data.add(M, self.scc[i][i]._call(t), (dy[i]**2-dt)/2)
            for j in range(i):
                M = _data.add(M, self.scc[i][j]._call(t), dy[i]*dy[j])
        out = _data.matmul(M, state)
        if self._issuper:
            Mdag = M.adjoint()
            out = _data.matmul(out, Mdag)
            for cop in self.c_ops:
                op = cop._call(t)
                out += op @ state @ op.adjoint() * dt
            out = out / _data.trace(out)
        else:
            out = out / _data.norm.l2(out)
        return out

    @property
    def options(self):
        """
        Supported options by Rouchon Stochastic Integrators:

        dt : float, default=0.001
            Internal time step.

        tol : float, default=1e-7
            Relative tolerance.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


StochasticSolver.add_integrator(RouchonSODE, "rouchon")
