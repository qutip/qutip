

import numpy as np
import matplotlib.pyplot as plt
from qutip import mcsolve, sesolve, mesolve, steadystate
from qutip import Options, BaseProgressBar


class Qsystem:
    def __init__(self, H, c_ops=[], args={}, e_ops=[]):
        # system
        self.H = H
        self.c_ops = c_ops
        self.args = args
        self.e_ops = e_ops
        self.L = None

        # state
        self.t = 0.
        self.t0 = 0.
        self.state_0 = None
        self.state_t = None
        self.states_runs_t = []
        self._tlist = []

        # options
        self.progress_bar = BaseProgressBar()
        self._options = Options()
        self.ntraj = [1]

        # self.optimization
        self.period = 0
        self.diagonalized = False
        self.sparsity = 1

    def reset(self, t=0, state=None, tlist=[]):
        if tlist:
            self._tlist = _tlist
            self.t0 = _tlist[0]
            self.t = _tlist[0]
        else:
            self.t0 = t
            self.t = self.t0

        self._check_consistance(state)
        self.state_0 = state
        self.state_t = None
        self.states_runs_t = []




    def evolve(self, tlist=[], state0=None, t=None):
        if tlist:
            self.tlist = tlist
        if not self._tlist:
            smart_tlist = True
        else:
            smart_tlist = False

        if state0 is None:
            state0 = self.state_0

        if self.ready:
            self.prepare()

        if self.evolve_type == "me":
            # use mesolve
            res = mesolve(self.H, state0, tlist, self.c_ops,
                          e_ops=self._e_ops,
                          args=self.args, options=self.options,
                          progress_bar=self.progress_bar)

        elif self.evolve_type == "se":
            # use sesolve
            res = sesolve(self.H, state0, tlist, e_ops=self._e_ops,
                          args=self.args, options=self.options,
                          progress_bar=self.progress_bar)

        elif self.evolve_type == "mc":
            # use sesolve
            res = mcsolve(self.H, state0, tlist, self.c_ops,
                          e_ops=self._e_ops, args=self.args, ntraj=self.ntraj,
                          options=self.options,
                          progress_bar=self.progress_bar)

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

        res = mcsolve(H, state0, tlist, c_ops=self.c_ops, e_ops=self._e_ops,
                      ntraj=ntraj, args=self.args, options=self.options,
                      progress_bar=self.progress_bar)
        self.state_t = res.final_state
        self.t = tlist[-1]
        self.path_results = res

    def steadystate(method='direct', solver=None, **kwargs):
        ss = steadystate(self.H, self.c_ops, method=method, solver=solver,
                         **kwargs)
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

    def states(self):
        self.path_results.expect

    def runs_expect(self):
        self.path_results.expect

    def runs_expect_averages(self):
        self.path_results.expect

    def runs_states(self):
        self.path_results.expect

    def final_state(self):
        pass

    def runs_final_states(self):
        pass
