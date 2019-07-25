




import numpy as np
import matplotlib.pyplot as plt
from qutip import mcsolve, sesolve, mesolve, steadystate,
from qutip import Options, BaseProgressBar





class Qsystem:
    def __init__(self, H, c_ops=[], args={}):
        self.H = H
        self.c_ops = c_ops
        self.args = args

        self._e_ops = []
        self.t = 0
        self.state_0 = None
        self.state_t = None
        self.progress_bar = BaseProgressBar()

        self._options = Options()

    def reset(self, t, state=None):
        self.t = t
        self.state_0 = state
        self.state_t = state

    @property
    def e_ops(self):
        return self._e_ops

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, other):
        if isinstance(other Options):
            self._options = other
        else:
            raise Exception("options must be an qutip.Options instance")

    def evolve(self, tlist=[], state0=None, t=None, dt=None):
        if not np.arraylike(tlist):
            if t is not None and dt is not None:
                tlist = np.linspace(selt.t, t, (t-self.t)//dt+1)
                if state0 is None and self.state_t is not None:
                    state0 = self.state_t
            else:
                raise TypeError("tlist expected")
        if state0 is None:
            state0 = self.state_0
        if c_ops:
            # use mesolve
            res = mesolve(self.H, state0, tlist, self.c_ops, self._e_ops,
                          args=self.args,
                          options=self.options, progress_bar=self.progress_bar)
        else:
            # use sesolve
            res = sesolve(self.H, state0, tlist, e_ops=self._e_ops,
                          args=self.args,
                          options=self.options, progress_bar=self.progress_bar)
        self.state_t = res.final_state
        self.t = tlist[-1]
        self.evolution_results = res

    def get_trajectories(self, ntraj=100, tlist=[], state0=None, t=None, dt=None):
        if not np.arraylike(tlist):
            if t is not None and dt is not None:
                tlist = np.linspace(selt.t, t, (t-self.t)//dt+1)
                if state0 is None and self.state_t is not None:
                    state0 = self.state_t
            else:
                raise TypeError("tlist expected")
        if state0 is None:
            state0 = self.state_0

        res = mcsolve(H, state0, tlist, c_ops=self.c_ops, e_ops=self._e_ops, ntraj=ntraj,
                      args=self.args, options=self.options, progress_bar=self.progress_bar)
        self.state_t = res.final_state
        self.t = tlist[-1]
        self.path_results = res

    def steadystate(method='direct', solver=None, **kwargs):
        ss = steadystate(self.H, self.c_ops, method=method, solver=solver, **kwargs)
        self.ss = ss

    def plot_expect(self):
        if self.evolution_results:
            plot_expectation_value(self.evolution_results)
        if self.path_results:
            plot_expectation_value(self.path_results)
        if self.ss:
            expect_fs = []
            for e_op in self.e_ops:
                expect_fs.append(expect(e_op, self.ss))
                plt.axhline(y=expect_fs[-1], lw=1.5)

    def optimization(self, period=0, diagonalize=False, sparse_state=False):
        raise NotImplementedError

    def expect(self):
        self.evolution_results.expect

    def runs_expect(self):
        self.path_results.expect

    def runs_expect_averages(self):
        self.path_results.expect
