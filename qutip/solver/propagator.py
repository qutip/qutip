__all__ = ['propagator', 'propagator_steadystate']

import numbers
import numpy as np

from .. import Qobj, qeye, unstack_columns, QobjEvo
from ..core import data as _data
from .mesolve import mesolve
from .sesolve import sesolve
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

    t : float or array-like
        Time or list of times for which to evaluate the propagator.

    c_op : list, optional
        List of Qobj or QobjEvo collapse operators.

    args : list/array/dictionary
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
