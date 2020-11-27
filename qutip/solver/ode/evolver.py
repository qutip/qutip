
from ..evolver import Evolver, evolver_collection
from .verner7efficient import vern7
from .verner9efficient import vern9
from .wrapper import QtOdeFuncWrapperSolverQEvo

class EvolverVern(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use verner method implimented in cython
    #
    _error_msg = ("ODE integration error: Try to increase "
                  "the allowed number of substeps by increasing "
                  "the nsteps parameter in the Options class.")
    description = "qutip implementation of verner most efficient runge-kutta"
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'min_step', 'interpolate']

    def prepare(self):
        func = QtOdeFuncWrapper(self.system)
        opt = {key: self.options[key]
               for key in self.used_options
               if key in self.options}
        ode = vern7 if self.options['method'] == 'vern7' else vern9
        self._ode_solver = ode(func, **opt)
        self.name = "qutip " + self.options['method']

    def get_state(self, copy=True):
        state = self._ode_solver.y
        return self._ode_solver.t, state.copy() if copy else state

    def set_state(self, state, t):
        self._ode_solver.set_initial_value(state, t)

    def backstep(self, t):
        self._ode_solver.integrate(t)
        return self.get_state()

    def step(self, t):
        """ Evolve to t, must be `set` before. """
        self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def one_step(self, t):
        """ Evolve to t, must be `set` before. """
        self._ode_solver.integrate(t, step=True)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()


vernlimits = {
    "base": True,
    "backstep": True,
    "update_args": True,
    "feedback": True,
    "cte": False,
}

evolver_collection.add(EvolverVern, methods=['vern7', 'vern9'],
                       limits=vernlimits, _test=False)


class EvolverDiag(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Diagonalize the Hamiltonian and
    # This should be used for constant Hamiltonian evolution (sesolve, mcsolve)
    #
    description = "Diagonalize a constant system"
    used_options = []

    def __init__(self, system, options, args, feedback_args):
        if not system.const:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        self.system = system
        self.options = options
        self._dt = 0.
        self._expH = None

    def prepare(self):
        self.diag, self.U = system(0).eigenstates()
        self.diag = self.diag.reshape((-1,1))
        self.U = np.hstack([eket.full() for eket in self.U])
        self.Uinv = np.linalg.inv(self.U)
        self.name = "qutip diagonalized"

    def step(self, t):
        """ Evolve to t, must be `set` before. """
        dt = t - self._t
        if dt == 0:
            return self.get_state()
        elif self._dt != dt:
            self._expH = np.exp(self.diag * dt)
            self._dt = dt
        self._y *= self._expH
        self._t = t
        return self.get_state()

    def one_step(self, t):
        return self.step(t)

    def backstep(self, t):
        return self.step(t)

    def get_state(self, copy=False):
        y = self.U @ self._y
        return self._t, _data.dense.fast_from_numpy(y)

    def set_state(self, state0, t):
        self._t = t
        self._y = (self.Uinv @ state0.to_array())

diaglimits = {
    "base": True,
    "backstep": True,
    "update_args": False,
    "feedback": False,
    "cte": True,
}

evolver_collection.add(EvolverDiag, methods=['diag'],
                       limits=diaglimits, _test=False)
