__all__ = ['Propagator', 'propagator', 'propagator_steadystate']

import numbers
import numpy as np

from .. import Qobj, qeye, unstack_columns, QobjEvo
from ..core import data as _data
from .mesolve import mesolve, MeSolver
from .sesolve import sesolve, SeSolver
from .options import SolverOptions


def propagator(H, t, c_ops=(), args=None, options=None):
    r"""
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\rho_{\mathrm vec}(t) = U(t) \rho_{\mathrm vec}(0)`
    where :math:`\rho_{\mathrm vec}` is the vector representation of the
    density matrix.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. ``list`` of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    t : float or array-like, optional
        Time or list of times for which to evaluate the propagator.

    c_ops : list, optional
        List of Qobj or QobjEvo collapse operators.

    args : dictionary
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.SolverOptions`
        Options for the ODE solver.

    Returns
    -------
    U : qobj, list
        Instance representing the propagator(s) :math:`U(t)`. Return a single
        Qobj when ``t`` is a number or a list when ``t`` is a list.

    """
    if isinstance(t, numbers.Real):
        tlist = [0, t]
        list_output = False
    else:
        tlist = t
        list_output = True

    if not isinstance(H, (Qobj, QobjEvo)):
        H = QobjEvo(H, args=args, tlist=tlist)

    if H.issuper or c_ops:
        out = mesolve(H, qeye(H.dims), tlist, c_ops=c_ops,
                      args=args, options=options).states
    else:
        out = sesolve(H, qeye(H.dims[0]), tlist,
                      args=args, options=options).states

    if list_output:
        return out
    else:
        return out[-1]


def propagator_steadystate(U):
    r"""Find the steady state for successive applications of the propagator
    :math:`U`.

    Parameters
    ----------
    U : qobj
        Operator representing the propagator.

    Returns
    -------
    a : qobj
        Instance representing the steady-state density matrix.
    """
    evals, estates = U.eigenstates()
    shifted_vals = np.abs(evals - 1.0)
    ev_idx = np.argmin(shifted_vals)
    rho_data = unstack_columns(estates[ev_idx].data)
    rho_data = _data.mul(rho_data, 0.5 / _data.trace(rho_data))
    return Qobj(_data.add(rho_data, _data.adjoint(rho_data)),
                dims=U.dims[0],
                type='oper',
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
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. ``list`` of [:class:`Qobj`, :class:`Coefficient`] or callable
        that can be made into :class:`QobjEvo` are also accepted.

    c_ops : list, optional
        List of Qobj or QobjEvo collapse operators.

    args : dictionary
        Parameters to callback functions for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.SolverOptions`
        Options for the ODE solver.

    memoize : int [10]
        Max number of propagator to save.

    tol : float [1e-14]
        Absolute tolerance for the time. If a previous propagator was computed
        at a time within tolerance, that propagator will be returned.

    .. note :
        The :class:`Propagator` is not a :class:`QobjEvo` so it cannot be used
        for operations with :class:`Qobj` or :class:`QobjEvo`. It can be made
        into a :class:`QobjEvo` with ::
            U = QobjEvo(Propagator(H))
    """
    def __init__(self, H, c_ops=(), args=None, options=None,
                 memoize=10, tol=1e-14):
        Hevo = QobjEvo(H, args=args)
        c_ops = [QobjEvo(op, args=args) for op in c_ops]
        self.times = [0]
        self.invs = [None]
        if Hevo.issuper or c_ops:
            self.props = [qeye(Hevo.dims)]
            self.solver = MeSolver(Hevo, c_ops=c_ops, options=options)
        else:
            self.props = [qeye(Hevo.dims[0])]
            self.solver = SeSolver(Hevo, options=options)
        self.cte = self.solver.rhs.isconstant
        self.unitary = (not self.solver.rhs.issuper
                        and isinstance(H, Qobj)
                        and H.isherm)
        self.args = args
        self.memoize = max(3, memoize)
        self.tol = tol

    def __call__(self, t, t_start=0, **args):
        """
        Get the propagator at ``t``.

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
            self.times = [0]
            self.props = [qeye(self.props[0].dims[0])]

        if t_start:
            U = self._prop2t(t, t_start)
        else:
            idx = np.searchsorted(self.times, t)
            if idx < len(self.times) and abs(t-self.times[idx]) <= self.tol:
                U = self.props[idx]
            elif idx > 0 and abs(t-self.times[idx-1]) <= self.tol:
                U = self.props[idx-1]
            else:
                U = self._compute(t, idx)
                self._insert(t, U, idx)
        return U

    def inv(self, t, **args):
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

    def _prop2t(self, t_end, t_start):
        """
        Obtain the probagator between times.
            ``psi[t] = U.prop(t, t_start) @ psi[t_start]``
        """
        if t_end == t_start:
            return self(0)
        if self.cte:
            return self(t_end - t_start)
        else:
            return self(t_end) @ self.inv(t_start)

    def _compute(self, t, idx):
        """
        Compute the propagator at ``t``, ``idx`` point to a pair of
        (time, propagator) close to the desired time.
        """
        if idx > 0:
            self.solver.start(self.props[idx-1], self.times[idx-1])
            U = self.solver.step(t, args=self.args)
        else:
            # Evolving backward in time is not supported by all integrator.
            self.solver.start(qeye(self.props[0].dims[0]), t)
            Uinv = self.solver.step(self.times[idx], args=self.args)
            U = self._inv(Uinv)
        return U

    def _inv(self, U):
        return U.dag() if self.unitary else U.inv()

    def _insert(self, t, U, idx):
        """
        Insert a new pair of (time, propagator) to the memorized states.
        """
        self.times = self.times[:idx] + [t] + self.times[idx:]
        self.props = self.props[:idx] + [U] + self.props[idx:]
        if len(self.times) > self.memoize:
            # When the list get too long, we clean memory.
            # We keep the extremums and last call.
            # There are probably a good ways to do this.
            rm_idx = np.random.randint(1, self.memoize-1)
            rm_idx = rm_idx if rm_idx < idx else rm_idx + 1
            self.times = self.times[:rm_idx] + self.times[rm_idx+1:]
            self.props = self.props[:rm_idx] + self.props[rm_idx+1:]
