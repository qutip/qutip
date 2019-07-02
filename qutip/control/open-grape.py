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
"""
This module contains functions that implement the GRAPE algorithm for
calculating pulse sequences for open quantum systems.
"""
 
__all__ = ['plot_grape_control_fields',
           'open_q_grape_unitary',]
 
import warnings
# import time
import numpy as np
from scipy.interpolate import interp1d
 
# from qutip.qobj import Qobj
from qutip.operators import identity
from qutip.ui.progressbar import BaseProgressBar
from qutip.tensor import tensor

 
# import qutip.logging_utils
# logger = qutip.logging_utils.get_logger()
 
 
class GRAPEResult:
    """
    Class for representing the result of a GRAPE simulation.
 
    Attributes
    ----------
    u : array
        GRAPE control pulse matrix.
 
    H_t : time-dependent Hamiltonian
        The time-dependent Hamiltonian that realize the GRAPE pulse sequence.
 
    U_f : Qobj
        The final unitary transformation that is realized by the evolution
        of the system with the GRAPE generated pulse sequences.
 
    """
    def __init__(self, u=None, H_t=None, U_f=None):
 
        self.u = u
        self.H_t = H_t
        self.U_f = U_f
 
 
def plot_grape_control_fields(times, u, labels, uniform_axes=False):
    """
    Plot a series of plots showing the GRAPE control fields given in the
    given control pulse matrix u.
 
    Parameters
    ----------
    times : array
        Time coordinate array.
 
    u : array
        Control pulse matrix.
 
    labels : list
        List of labels for each control pulse sequence in the control pulse
        matrix.
 
    uniform_axes : bool
        Whether or not to plot all pulse sequences using the same y-axis scale.
 
    """
    import matplotlib.pyplot as plt
 
    R, J, M = u.shape
 
    fig, axes = plt.subplots(J, 1, figsize=(8, 2 * J), squeeze=False)
 
    y_max = abs(u).max()
 
    for r in range(R):
        for j in range(J):
 
            if r == R - 1:
                lw, lc, alpha = 2.0, 'k', 1.0
 
                axes[j, 0].set_ylabel(labels[j], fontsize=18)
                axes[j, 0].set_xlabel(r'$t$', fontsize=18)
                axes[j, 0].set_xlim(0, times[-1])
 
            else:
                lw, lc, alpha = 0.5, 'b', 0.25
 
            axes[j, 0].step(times, u[r, j, :], lw=lw, color=lc, alpha=alpha)
 
            if uniform_axes:
                axes[j, 0].set_ylim(-y_max, y_max)
 
    fig.tight_layout()
 
    return fig, axes
 
 
def _overlap(A, B):
    return (A.dag() * B).tr() / A.shape[0]
    # return cy_overlap(A.data, B.data)
 
 

def open_q_grape_unitary(U, H0, H_ops, c_ops, R, times, eps=None,
                         u_start=None,
                         u_limits=None, interp_kind='linear',
                         use_interp=False,
                         alpha=None, beta=None, phase_sensitive=True,
                         progress_bar=BaseProgressBar()):
    """
    Calculate control pulses for the Hamiltonian operators in H_ops so that the
    unitary U is realized.
 
    Experimental: Work in progress.
 
    Parameters
    ----------
    U : Qobj
        Target unitary evolution operator.
 
    H0 : Qobj
        Static Hamiltonian (that cannot be tuned by the control fields).
 
    H_ops: list of Qobj
        A list of operators that can be tuned in the Hamiltonian via the
        control fields.
 
    c_ops : list of Qobj
            List of collapse operators. Any common coefficient
            like gamma must be multiplied in before c_ops is passed.
 
    R : int
        Number of GRAPE iterations.
 
    time : array / list
        Array of time coordinates for control pulse evalutation.
 
    u_start : array
        Optional array with initial control pulse values.
 
    Returns
    -------
        Instance of GRAPEResult, which contains the control pulses calculated
        with GRAPE, a time-dependent Hamiltonian that is defined by the
        control pulses, as well as the resulting propagator.
 
    """
 
    if eps is None:
        eps = 0.1 * (2 * np.pi) / (times[-1])
 
    UT_liou = tensor(U.trans(), U)
    M = len(times)
    J = len(H_ops)
 
    u = np.zeros((R, J, M))
    '''
    if H0 is not None: # Useless
        pass
 
    else:
        raise ValueError("H0 is zero ")
    '''
 
    # Making the Identity matrix
    # II = identity(H0.shape[0])
    # Causes dims errors
    Idims = H0.dims[0]
    II_parts = [identity(di) for di in Idims]
    II = tensor(II_parts)
 
    # Shape Checker
    # 1. Shape checking within H
 
    # H0_size = H0.shape[0]
    H0_dims = H0.dims[0]
    for jj in range(len(H_ops)):
        if H_ops[jj].dims[0] == H0_dims:
            # print("ok", jj)
            continue
 
        else:
            print('Failed for {0} th H_ops {1}'.format(jj, H_ops[jj]))
            raise ValueError(" All Hamiltonians must be of the same size")
 
    if c_ops:
        # 2. Shape checking within c_ops
        c_ops_dims = c_ops[0].dims[0]
        for ckkk in c_ops:
            if ckkk.dims[0] == c_ops_dims:
                # print("ok", kkk) # Not checked yet.
                continue
 
            else:
                print('Failed for {0} th c_ops {1}'.format(kkk, c_ops[kkk]))
                raise ValueError("All c_ops must be of the same size")
 
        # 3. Shape checking betweeh c_ops and H
        if c_ops_dims == H0_dims:
            print("c_ops compatible with H ")  # Not checked yet.
            # All sizes are ok.
            pass
 
        else:
            print("c_ops[0].shape[0] is not equal to H0.shape[0]")
            raise ValueError("Hamiltonian shapes incompatible with c_ops")
 
    if u_limits and len(u_limits) != 2:
        raise ValueError("u_limits must be a list with two values")
 
    if u_limits:
        warnings.warn("Caution: Using experimental feature u_limits")
 
    if u_limits and u_start:
        # make sure that no values in u0 violates the u_limits conditions
        u_start = np.array(u_start)
        u_start[u_start < u_limits[0]] = u_limits[0]
        u_start[u_start > u_limits[1]] = u_limits[1]
 
    if u_start is not None:
        for idx, u0 in enumerate(u_start):
            u[0, idx, :] = u0
 
    if beta:
        warnings.warn("Causion: Using experimental feature time-penalty")
 
    def x_m(j):
        H_j = H_ops[j]
        x_m_j = tensor(II, H_j) - tensor(H_j.conj(), II)
        return x_m_j
 
    progress_bar.start(R)
    for r in range(R - 1):
        progress_bar.update(r)
 
        dt = times[1] - times[0]
 
        if use_interp:
            ip_funcs = [interp1d(times, u[r, j, :], kind=interp_kind,
                                 bounds_error=False, fill_value=u[r, j, -1])
                        for j in range(J)]
 
            def _H_t(t, args=None):
                return H0 + sum([float(ip_funcs[j](t)) * H_ops[j]
                                 for j in range(J)])
 
            def _A_t(t, args=None):
                H_conj = _H_t(t).conj()
                H = _H_t(t)
                ham_part = tensor(II, H) - tensor(H_conj, II)
                # tensor(II, _H_t(t)) - tensor(H_conj, II)
 
                # Preparing the c_ops part
                if c_ops:
                    c_ops_full = 0
                    for ckk in c_ops:
                        c1 = tensor((ckk.trans()), ckk)
                        c2 = tensor(II, ((ckk.dag())*(ckk)))
                        c3 = tensor(((ckk.trans())*(ckk.conj())), II)
                        c_add_real = c1 - 0.5*(c2 + c3)
                        c_ops_full = c_ops_full + 1j*c_add_real
 
                    A = ham_part + c_ops_full
                else:
                    A = ham_part
 
                return A
 
            U_list = [(-1j * _A_t(times[idx]) * dt).expm()
                      for idx in range(M-1)]
 
        else:
            def _H_idx(idx):
                return H0 + sum([u[r, j, idx] * H_ops[j] for j in range(J)])
 
            def _A_idx(idx):
                H_conj = _H_idx(idx).conj()
                H = _H_idx(idx)
                ham_part = tensor(II, H) - tensor(H_conj, II)
 
                # Preparing the c_ops part
                if c_ops:
                    c_ops_full = 0
                    for ckk in c_ops:
                        c1 = tensor(ckk.trans(), ckk)
                        c2 = tensor(II, ckk.dag()*ckk)
                        c3 = tensor(ckk.trans(), ckk.conj())
                        c_add_real = c1 - 0.5*(c2 + c3)
                        c_ops_full = c_ops_full + 1j*c_add_real
 
                    A = ham_part + c_ops_full
 
                else:
                    A = ham_part
 
                return A
 
            U_list = [(-1j * _A_idx(idx) * dt).expm() for idx in range(M-1)]
 
        U_f_list = []
        U_b_list = []
 
        U_f = 1
        U_b = 1
        for n in range(M - 1):
 
            U_f = U_list[n] * U_f
            U_f_list.append(U_f)
 
            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - n].dag() * U_b
 
        '''
        U_store_name = "U-store" + str(r) + ".txt"
        with open(U_store_name, 'w') as f:
            f.write('U_list\n')
            f.write(str(U_list))
            f.write('U_f_list\n')
            f.write(str(U_f_list))
            f.write('U_b_list\n')
            f.write(str(U_b_list))
        '''
 
        for j in range(J):
            for m in range(M-1):
                P = U_b_list[m] * UT_liou
                Q = 1j * dt * x_m(j) * U_f_list[m]
 
                if phase_sensitive:
                    du = - _overlap(P, Q)
                else:
                    du = - 2 * _overlap(P, Q) * _overlap(U_f_list[m], P)
 
                if alpha:
                    # penalty term for high power control signals u
                    du += -2 * alpha * u[r, j, m] * dt
 
                if beta:
                    # penalty term for late control signals u
                    du += -2 * beta * m * u[r, j, m] * dt
 
                u[r + 1, j, m] = u[r, j, m] + eps * du.real
 
                if u_limits:
                    if u[r + 1, j, m] < u_limits[0]:
                        u[r + 1, j, m] = u_limits[0]
                    elif u[r + 1, j, m] > u_limits[1]:
                        u[r + 1, j, m] = u_limits[1]
 
            u[r + 1, j, -1] = u[r + 1, j, -2]
 
    if use_interp:
        ip_funcs = [interp1d(times, u[R - 1, j, :], kind=interp_kind,
                             bounds_error=False, fill_value=u[R - 1, j, -1])
                    for j in range(J)]
 
        H_td_func = [H0] + [[H_ops[j], lambda t, args, j=j: ip_funcs[j](t)]
                            for j in range(J)]
    else:
        H_td_func = [H0] + [[H_ops[j], u[-1, j, :]] for j in range(J)]
 
    progress_bar.finished()
 
    # return U_f_list[-1], H_td_func, u
    return GRAPEResult(u=u, U_f=U_f_list[-1], H_t=H_td_func)
 
