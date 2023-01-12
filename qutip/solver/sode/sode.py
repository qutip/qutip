
from . import _sode as sstepper
from ..integrator import SIntegrator


class ExplicitIntegrator(SIntegrator):
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-7,
    }
    def __init__(self, system, options):
        self.system = system
        self.options = options
        self.dt = self.options["dt"]
        self.tol = self.options["tol"]
        self.stepper = getattr(sstepper, options["method"])
        self.N_dw = sstepper.N_dws[options["method"]]
        self.N_drift = system.num_collapse

    def _step(self, dt, dW):
        new_state = self.stepper(self.system, self.t, self.state, dt, dW)
        self.state = new_state
        self.t += dt

    def set_state(self, t, state0, generator):
        self.t = t
        self.state = state0
        self.generator = generator

    def integrate(self, t, copy=True):
        delta_t = (t - self.t)
        dt = self.dt
        N, extra = np.divmod(delta_t, dt)
        if extra > self.tol:
            # Not a whole number of steps.
            N += 1
            dt = delta_t / N
        dW = self.generator.normal(0, np.sqrt(dt), size=(N, self.N_drift, self.N_dw))

        for i in range(N):
            self._step(dt, dW[i, :])

        return self.t, self.state, np.sum(dW, axis=0) / (N * dt)

    def get_state(self, copy=True):
        return self.t, self.state, self.generator
