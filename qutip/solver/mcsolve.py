# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['mcsolve', "McSolver", "MeMcSolver"]

import warnings

import numpy as np
import numpy.random as np_rng
from numpy.random import Generator, SeedSequence
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from ..core import (Qobj, QobjEvo, spre, spost, liouvillian, isket, ket2dm,
                    stack_columns, unstack_columns)
from ..core.data import to
from ..core import data as _data
from .options import SolverOptions
from .result import Result, MultiTrajResult, MultiTrajResultAveraged
from .solver_base import Solver, _to_qevo
from .sesolve import sesolve
from .mesolve import mesolve
from .evolver import *
from time import time


def mcsolve(H, psi0, tlist, c_ops=None, e_ops=None, ntraj=1,
            args=None, options=None, seeds=None):
    r"""Monte Carlo evolution of a state vector :math:`|\psi \rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Options class.

    mcsolve supports time-dependent Hamiltonians and collapse operators using
    either Python functions of strings to represent time-dependent
    coefficients. Note that, the system Hamiltonian MUST have at least one
    constant term.

    As an example of a time-dependent problem, consider a Hamiltonian with two
    terms ``H0`` and ``H1``, where ``H1`` is time-dependent with coefficient
    ``sin(w*t)``, and collapse operators ``C0`` and ``C1``, where ``C1`` is
    time-dependent with coeffcient ``exp(-a*t)``.  Here, w and a are constant
    arguments with values ``W`` and ``A``.

    Using the Python function time-dependent format requires two Python
    functions, one for each collapse coefficient. Therefore, this problem could
    be expressed as::

        def H1_coeff(t,args):
            return sin(args['w']*t)

        def C1_coeff(t,args):
            return exp(-args['a']*t)

        H = [H0, [H1, H1_coeff]]

        c_ops = [C0, [C1, C1_coeff]]

        args={'a': A, 'w': W}

    or in String (Cython) format we could write::

        H = [H0, [H1, 'sin(w*t)']]

        c_ops = [C0, [C1, 'exp(-a*t)']]

        args={'a': A, 'w': W}

    Constant terms are preferably placed first in the Hamiltonian and collapse
    operator lists.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, ``list``
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Number of trajectories to run.

    c_ops : :class:`qutip.Qobj`, ``list``
        single collapse operator or a ``list`` of collapse operators.

    e_ops : :class:`qutip.Qobj`, ``list``
        single operator as Qobj or ``list`` or equivalent of Qobj operators
        for calculating expectation values.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : SolverOptions
        Instance of ODE solver options.

    Returns
    -------
    results : :class:`qutip.solver.Result`
        Object storing all results from the simulation.

    .. note::

        It is possible to reuse the random number seeds from a previous run
        of the mcsolver by passing the output Result object seeds via the
        Options class, i.e. SolverOptions(seeds=prev_result.seeds).
    """
    args = args or {}
    feedback_args = feedback_args or {}
    options = options if options is not None else SolverOptions()

    # set the physics
    if not psi0.isket:
        raise ValueError("Initial state must be a state vector.")

    if c_ops is None:
        c_ops = []
    if len(c_ops) == 0:
        warnings.warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, _safe_mode=_safe_mode)

    # load monte carlo class
    mc = McSolver(H, c_ops, e_ops, options, tlist,
                  args, feedback_args, _safe_mode)
    if seeds is not None:
        mc.seed(ntraj, seeds)

    # Run the simulation
    return mc.run(psi0, tlist=tlist, ntraj=ntraj)




# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class McSolver(MultiTrajSolver):
    """
    ... TODO
    """
    _traj_solver_class = _OneTrajMcSolver
    _super = False
    name = "mcsolve"
    optionsclass = SolverOptions

    def __init__(self, H, c_ops, e_ops=None, options=None,
                 times=None, args=None, seed=None):
        _time_start = time()
        self.stats = {}
        if not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")

        self.seed_sequence = SeedSequence(seed)
        self.traj_solvers = []
        self.result = None

        self.e_ops = e_ops or []
        self.options = options

        self._c_ops = [QobjEvo(op, args=args, tlist=times) for c_op in c_ops]
        self._n_ops = [c_op.dag() * c_op for c_op in self._c_ops]
        self._system = -1j* QobjEvo(H, args=args, tlist=times)
        for n_evo in self._n_ops:
            self._system -= 0.5 * n_evo

        self.stats['solver'] = "MonteCarlo Evolution"
        self.stats['num_collapse'] = len(c_ops)
        self.stats["preparation time"] = time() - _time_start

    def _check_state_dims(self, state):
        if not state.isket:
            raise TypeError("The unitary solver requires psi0 to be "
                            "a ket as initial state "
                            "or a unitary as initial operator.")

        if self._system.dims[1] != state.dims[0]:
            raise TypeError("".join([
                            "incompatible dimensions ",
                            repr(self._system.dims),
                            " and ",
                            repr(state.dims),])
                           )



class _OneTrajMcSolver(_OneTraj):
    """
    ... TODO
    """
    _super = False
    name = "mcsolve"

    def __init__(self, rhs, *, e_ops=None, options=None):
        if isinstance(rhs, (QobjEvo, Qobj)):
            self.rhs = QobjEvo(rhs)
        else:
            TypeError("The rhs must be a QobjEvo")
        self.e_ops = e_ops
        self.options = options
        self.stats = {"preparation time": 0}
        self._state_metadata = {}

        if options.mcsolve['BitGenerator']:
            if hasattr(np_rng, options.mcsolve['BitGenerator']):
                self.bit_gen = getattr(np_rng, options.mcsolve['BitGenerator'])
            else:
                raise ValueError("BitGenerator is not know to numpy.random")
        else:
            self.bit_gen = np_rng.PCG64

    def run(self, state0, tlist, *,
            args=None, e_ops=None, options=None, seed=None):
        self.collapses = []
        result = super().run(state0, tlist, args=args,
                             e_ops=e_ops, options=options, seed=seed)
        result.collapse = list(self.collapses)
        return result

    def prob_func(self, state):
        return _data.norm.l2(state)**2

    def _step(self, t, copy=True):
        """Evolve to t, including jumps."""
        t_old, y_old = self.get_state(copy=False)
        norm_old = self.prob_func(y_old)
        while t_old < t:
            t_step, state = self._integrator.mcstep(t, copy=False)
            norm = self.prob_func(state)
            if norm <= self.target_norm:
                self._do_collapse(norm_old, norm, t_old)
                t_old, y_old = self.get_state(copy=False)
                norm_old = self.prob_func(y_old)
            else:
                t_old = t_step
                norm_old = norm
                y_old = state

        return t_old, _data.mul(y_old, 1 / self.norm_func(y_old))

    def _do_collapse(self, norm_old, norm, t_prev):
        t_final, _ = self._integrator.get_state()
        tries = 0
        while tries < self.norm_steps:
            tries += 1
            if (t_final - t_prev) < self.norm_t_tol:
                t_guess = t_final
                _, state = self._integrator.get_state()
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
            if (t_guess - t_prev) < self.norm_t_tol:
                t_guess = t_prev + self.norm_t_tol
            _, state = self._integrator.mcstep(t, copy=False)
            norm2_guess = self.prob_func(state)
            if (
                np.abs(self.target_norm - norm2_guess) <
                self.norm_tol * self.target_norm
            ):
                break
            elif (norm2_guess < self.target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                norm_old = norm2_guess

        if tries >= self.norm_steps:
            raise Exception("Norm tolerance not reached. " +
                            "Increase accuracy of ODE solver or " +
                            "SolverOptions.mcsolve['norm_steps'].")

        # t_guess, state is at the collapse
        probs = np.zeros(len(self.n_ops))
        for i, n_op in enumerate(self.n_ops):
            probs[i] = n_op.expect(t, state, 1)
        probs = np.cumsum(probs)
        which = np.searchsorted(probs, probs[-1] * self.generator.random())

        state_new = self.c_ops[which].matmul_data(t_guess, state)
        new_norm = self.prob_func(state_new)
        if new_norm < self.mc_corr_eps:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self.norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((t_guess, which))
            self.target_norm = self.generator.random()
        self._integrator.set_state(t_guess, state_new)

    def _argument(self, args):
        if self._integrator is not None:
            self._integrator.arguments(args)
        else:
            self.rhs.arguments(args)
        for c_op in self._c_ops:
            c_op.arguments(args)
        for n_op in self._n_ops:
            n_op.arguments(args)
