# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ['Propagator', 'propagator', 'propagator_steadystate']

import numbers
import numpy as np

from .. import Qobj, qeye, qeye_like, unstack_columns, QobjEvo, liouvillian
from ..core import data as _data
from ..typing import QobjEvoLike
from .mesolve import mesolve, MESolver
from .sesolve import sesolve, SESolver
from .heom.bofin_solvers import HEOMSolver
from .solver_base import Solver
from .multitraj import MultiTrajSolver
from numbers import Number
from typing import Any


def propagator(
    H: QobjEvoLike,
    t: Number,
    c_ops: QobjEvoLike | list[QobjEvoLike] = None,
    args: dict[str, Any] = None,
    options: dict[str, Any] = None,
    **kwargs,
) -> Qobj | list[Qobj]:
    r"""
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\rho_{\mathrm vec}(t) = U(t) \rho_{\mathrm vec}(0)`
    where :math:`\rho_{\mathrm vec}` is the vector representation of the
    density matrix.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. ``list`` of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable
        that can be made into :obj:`.QobjEvo` are also accepted.

    t : float or array-like
        Time or list of times for which to evaluate the propagator. If a single
        time ``t`` is passed, the propagator from ``0`` to ``t`` is computed.
        When ``t`` is a list, the propagators from the first time in the list
        to each elements in ``t`` is returned. In that case, the first output
        will always be the identity matrix.

    c_ops : list, optional
        List of collapse operators as Qobj, QobjEvo or list that can be made
        into QobjEvo.

    args : dictionary, optional
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : dict, optional
        Options for the solver.

    **kwargs :
        Extra parameters to use when creating the
        :obj:`.QobjEvo` from a list format ``H``. The most common are ``tlist``
        and ``order`` for array-based time dependance. See :obj:`.QobjEvo` for
        the full list.

    Returns
    -------
    U : :obj:`.Qobj`, list
        Instance representing the propagator(s) :math:`U(t)`. Return a single
        Qobj when ``t`` is a number or a list when ``t`` is a list.

    Notes
    -----
    Unlike :func:`.sesolve` or :func:`.mesolve`, the output times in ``t`` are
    not used for array time dependent system. ``tlist`` must be passed as a
    keyword argument in those case. ``tlist`` and ``t`` can have different
    length and values.

    """
    if isinstance(t, numbers.Real):
        tlist = [0, t]
        list_output = False
    else:
        tlist = t
        list_output = True

    if not isinstance(H, (Qobj, QobjEvo)):
        H = QobjEvo(H, args=args, **kwargs)

    if isinstance(c_ops, list):
        c_ops = [QobjEvo(op, args=args, **kwargs) for op in c_ops]

    if c_ops:
        H = liouvillian(H, c_ops)

    U0 = qeye_like(H)

    if H.issuper:
        out = mesolve(H, U0, tlist, args=args, options=options).states
    else:
        out = sesolve(H, U0, tlist, args=args, options=options).states

    if list_output:
        return out
    else:
        return out[-1]


def propagator_steadystate(U: Qobj) -> Qobj:
    r"""Find the steady state for successive applications of the propagator
    :math:`U`.

    Parameters
    ----------
    U : :obj:`.Qobj`
        Operator representing the propagator.

    Returns
    -------
    a : :obj:`.Qobj`
        Instance representing the steady-state density matrix.
    """
    evals, estates = U.eigenstates()
    shifted_vals = np.abs(evals - 1.0)
    ev_idx = np.argmin(shifted_vals)
    rho_data = unstack_columns(estates[ev_idx].data)
    rho_data = _data.mul(rho_data, 0.5 / _data.trace(rho_data))
    return Qobj(_data.add(rho_data, _data.adjoint(rho_data)),
                dims=U.dims[0],
                isherm=True,
                copy=False)


class Propagator:
    """
    A generator of propagator for a system.

    Usage:

        U = Propagator(H, c_ops)

        psi_t = U(t) @ psi_0

    Save some previously computed propagator are stored to speed up subsequent
    computation. Changing ``args`` will erase these stored probagator.

    Parameters
    ----------
    system : :obj:`.Qobj`, :obj:`.QobjEvo`, :class:`.Solver`
        Possibly time-dependent system driving the evolution, either already
        packaged in a solver, such as :class:`.SESolver` or :class:`.BRSolver`,
        or the Liouvillian or Hamiltonian as a :obj:`.Qobj`,
        :obj:`.QobjEvo`. ``list`` of [:obj:`.Qobj`, :obj:`.Coefficient`]
        or callable that can be made into :obj:`.QobjEvo` are also accepted.

        Solvers that run non-deterministacilly, such as :class:`.MCSolver`, are
        not supported.

    c_ops : list, optional
        List of :obj:`.Qobj` or :obj:`.QobjEvo` collapse operators.

    args : dictionary, optional
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : dict, optional
        Options for the solver.

    memoize : int, default: 10
        Max number of propagator to save.

    tol : float, default: 1e-14
        Absolute tolerance for the time. If a previous propagator was computed
        at a time within tolerance, that propagator will be returned.

    Notes
    -----
    The :class:`Propagator` is not a :obj:`.QobjEvo` so
    it cannot be used for operations with :obj:`.Qobj` or
    :obj:`.QobjEvo`. It can be made into a
    :obj:`.QobjEvo` with ::

        U = QobjEvo(Propagator(H))

    """
    def __init__(
        self,
        system: Qobj | QobjEvo | Solver,
        *,
        c_ops: QobjEvoLike | list[QobjEvoLike] = None,
        args: dict[str, Any] = None,
        options: dict[str, Any] = None,
        memoize: int = 10,
        tol: float = 1e-14,
    ):
        if isinstance(system, MultiTrajSolver):
            raise TypeError("Non-deterministic solvers cannot be used "
                            "as a propagator system")
        elif isinstance(system, HEOMSolver):
            raise NotImplementedError(
                "HEOM is not supported by Propagator. "
                "Please, tell us on GitHub issues if you need it!"
            )
        elif isinstance(system, Solver):
            self.solver = system
        else:
            Hevo = QobjEvo(system, args=args)
            c_ops = c_ops if c_ops is not None else []
            c_ops = [QobjEvo(op, args=args) for op in c_ops]
            if Hevo.issuper or c_ops:
                self.solver = MESolver(Hevo, c_ops=c_ops, options=options)
            else:
                self.solver = SESolver(Hevo, options=options)

        self.times = [0]
        self.invs = [None]
        self.props = [qeye(self.solver.sys_dims)]
        self.solver.start(self.props[0], self.times[0])
        self.cte = self.solver.rhs.isconstant
        H_0 = self.solver.rhs(0)
        self.unitary = not H_0.issuper and H_0.isherm
        self.args = args
        self.memoize = max(3, int(memoize))
        self.tol = tol

    def _lookup_or_compute(self, t):
        """
        Get U(t) from cache or compute it.
        """
        idx = np.searchsorted(self.times, t)
        if idx < len(self.times) and abs(t-self.times[idx]) <= self.tol:
            U = self.props[idx]
        elif idx > 0 and abs(t-self.times[idx-1]) <= self.tol:
            U = self.props[idx-1]
        else:
            U = self._compute(t, idx)
            self._insert(t, U, idx)
        return U

    def __call__(self, t: float, t_start: float = 0, **args):
        """
        Get the propagator from ``t_start`` to ``t``.

        Parameters
        ----------
        t : float
            Time at which to compute the propagator.
        t_start: float [0]
            Time at which the propagator start such that:
                ``psi[t] = U.prop(t, t_start) @ psi[t_start]``
        args : dict
            Argument to pass to a time dependent Hamiltonian.
            Updating ``args`` take effect since ``t=0`` and the new ``args``
            will be used in future call.
        """
        # We could improve it when the system is constant using U(2t) = U(t)**2
        if not self.cte and args and args != self.args:
            self.args = args
            self.solver._argument(args)
            self.times = [0]
            self.props = [qeye_like(self.props[0])]
            self.solver.start(self.props[0], self.times[0])

        if t_start:
            if t == t_start:
                U = self._lookup_or_compute(0)
            if self.cte:
                U = self._lookup_or_compute(t - t_start)
            else:
                Uinv = self._inv(self._lookup_or_compute(t_start))
                U = self._lookup_or_compute(t) @ Uinv
        else:
            U = self._lookup_or_compute(t)
        return U

    def inv(self, t: float, **args):
        """
        Get the inverse of the propagator at ``t``, such that
            ``psi_0 = U.inv(t) @ psi_t``

        Parameters
        ----------
        t : float
            Time at which to compute the propagator.
        args : dict
            Argument to pass to a time dependent Hamiltonian.
            Updating ``args`` take effect since ``t=0`` and the new ``args``
            will be used in future call.
        """
        return self._inv(self(t, **args))

    def _compute(self, t, idx):
        """
        Compute the propagator at ``t``, ``idx`` point to a pair of
        (time, propagator) close to the desired time.
        """
        t_last = self.solver._integrator.get_state(copy=False)[0]
        if self.times[idx-1] <= t_last <= t:
            U = self.solver.step(t)
        elif idx > 0:
            self.solver.start(self.props[idx-1], self.times[idx-1])
            U = self.solver.step(t)
        else:
            # Evolving backward in time is not supported by all integrator.
            self.solver.start(qeye_like(self.props[0]), t)
            Uinv = self.solver.step(self.times[idx])
            U = self._inv(Uinv)
        return U

    def _inv(self, U):
        return U.dag() if self.unitary else U.inv()

    def _insert(self, t, U, idx):
        """
        Insert a new pair of (time, propagator) to the memorized states.
        """
        while len(self.times) >= self.memoize:
            rm_idx = self.memoize // 2
            if self.times[rm_idx] < t:
                idx -= 1
            del self.times[rm_idx]
            del self.props[rm_idx]
        self.times.insert(idx, t)
        self.props.insert(idx, U)
