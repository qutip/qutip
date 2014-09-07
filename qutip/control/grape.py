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
calculating pulse sequences for quantum systems.
"""
import time
import numpy as np
from scipy.interpolate import interp1d

from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.control.cy_grape import cy_overlap
from qutip.qip.gates import gate_sequence_product

def plot_grape_control_fields(times, u, labels, uniform_axes=False):
    """
    Plot a series of plots shoing the GRAPE control fields given in the
    matrix u.
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
    #return cy_overlap(A.data, B.data)

def overlap(A, B):
    return (A.dag() * B).tr() / A.shape[0]
    
def grape_unitary_orig(U, H0, H_ops, R, times, eps=None, u_start=None, interp_kind='linear', 
                  progress_bar=BaseProgressBar()):
    
    if eps is None:
        eps = 0.1 * (2 * pi) / (times[-1])

    M = len(times)
    J = len(H_ops)
    
    u = np.zeros((R, J, M))

    if u_start:
        for idx, u0 in enumerate(u_start):
            u[0, idx, :] = u0

    progress_bar.start(R)
    for r in range(R - 1):
        progress_bar.update(r)
        
        ip_funcs = [interp1d(times, u[r, j, :], kind=interp_kind, bounds_error=False, fill_value=u[r, j, -1])
                    for j in range(J)]

        def H_func(t, args=None):
            return sum([H0] + [float(ip_funcs[j](t)) * H_ops[j] for j in range(J)])    

        dt = np.diff(times)[0]

        U_list = [(-1j * H_func(times[idx]) * dt).expm() for idx in range(M-1)]

        U_f_list = []
        U_b_list = []

        U_f = 1
        U_b = 1
        for n in range(M - 1):

            U_f = U_list[n] * U_f
            U_f_list.append(U_f)

            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - n].dag() * U_b

        for j in range(J):
            for k in range(M - 1):
                u[r + 1, j, k] = u[r, j, k] - eps * overlap(U_b_list[k] * U, 1j * dt * H_ops[j] * U_f_list[k]).real

            u[r + 1, j, -1] = u[r + 1, j, -2]
            
    ip_funcs = [interp1d(times, u[R - 1, j, :], kind=interp_kind, bounds_error=False, fill_value=u[R - 1, j, -1])
                for j in range(J)]

    H_list_func = [H0] + [[H_ops[j], lambda t, args, j=j: ip_funcs[j](t)] for j in range(J)]    

    progress_bar.finished()
    
    return U_f_list[-1], H_list_func, u


def grape_unitary(U, H0, H_ops, R, times, eps=None, u_start=None,
                  u_limits=None, interp_kind='linear', use_interp=False,
                  alpha=None, phase_sensitive=True,
                  progress_bar=BaseProgressBar()):
    """
    Calculate control pulses for the Hamitonian operators in H_ops so that the
    unitary U is realized.

    Experimental: Work in progress.
    """

    if eps is None:
        eps = 0.1 * (2 * np.pi) / (times[-1])

    M = len(times)
    J = len(H_ops)
    
    u = np.zeros((R, J, M))

    if u_limits and len(u_limits) != 2:
        raise ValueError("u_limits must be a list with two values")

    if u_limits and u_start:
        # make sure that no values in u0 violates the u_limits conditions
        u_start = np.array(u_start)
        u_start[u_start < u_limits[0]] = u_limits[0]
        u_start[u_start > u_limits[1]] = u_limits[1]

    if u_start is not None:
        for idx, u0 in enumerate(u_start):
            u[0, idx, :] = u0

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

            U_list = [(-1j * _H_t(times[idx]) * dt).expm() for idx in range(M-1)]

        else:
            def _H_idx(idx):
                return H0 + sum([u[r, j, idx] * H_ops[j] for j in range(J)])    

            U_list = [(-1j * _H_idx(idx) * dt).expm() for idx in range(M-1)]

        U_f_list = []
        U_b_list = []

        U_f = 1
        U_b = 1
        for n in range(M - 1):

            U_f = U_list[n] * U_f
            U_f_list.append(U_f)

            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - n].dag() * U_b

        for j in range(J):
            for k in range(M-1):
                P = U_b_list[k] * U
                Q = 1j * dt * H_ops[j] * U_f_list[k]

    #            if phase_sensitive:
    #                du = - 2 * _overlap(P, Q) * _overlap(H_ops[j], P) 
    #            else:
                du = - _overlap(P, Q)

                if alpha:
                    # penalty term for high power control signals u
                    du += -2 * alpha * u[r, j, k] * dt

                u[r + 1, j, k] = u[r, j, k] + eps * du.real

                if u_limits:
                    if u_limits[0] < u[r + 1, j, k]:
                        u[r + 1, j, k] = u_limits[0]

                    elif u_limits[1] > u[r + 1, j, k]:
                        u[r + 1, j, k] = u_limits[1]

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
    
    return U_f_list[-1], H_td_func, u


def cy_grape_unitary(U, H0, H_ops, R, times, eps=None, u_start=None,
                  u_limits=None, interp_kind='linear', use_interp=False,
                  alpha=None, phase_sensitive=True,
                  progress_bar=BaseProgressBar()):
    """
    Calculate control pulses for the Hamitonian operators in H_ops so that the
    unitary U is realized.

    Experimental: Work in progress.
    """

    if eps is None:
        eps = 0.1 * (2 * np.pi) / (times[-1])

    M = len(times)
    J = len(H_ops)
    
    u = np.zeros((R, J, M))

    if u_limits and len(u_limits) != 2:
        raise ValueError("u_limits must be a list with two values")

    if u_limits and u_start:
        # make sure that no values in u0 violates the u_limits conditions
        u_start = np.array(u_start)
        u_start[u_start < u_limits[0]] = u_limits[0]
        u_start[u_start > u_limits[1]] = u_limits[1]

    if u_start is not None:
        for idx, u0 in enumerate(u_start):
            u[0, idx, :] = u0

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

            U_list = [(-1j * _H_t(times[idx]) * dt).expm() for idx in range(M-1)]

        else:
            def _H_idx(idx):
                return H0 + sum([u[r, j, idx] * H_ops[j] for j in range(J)])    

            U_list = [(-1j * _H_idx(idx) * dt).expm() for idx in range(M-1)]

        U_f_list = []
        U_b_list = []

        U_f = 1
        U_b = 1
        for n in range(M - 1):

            U_f = U_list[n] * U_f
            U_f_list.append(U_f)

            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - n].dag() * U_b

        for j in range(J):
            for k in range(M-1):
                P = U_b_list[k] * U
                Q = 1j * dt * H_ops[j] * U_f_list[k]

                #if phase_sensitive:
                #    du = - 2 * cy_overlap(P.data, Q.data) * cy_overlap(H_ops[j].data, P.data) 
                #else:
                du = - cy_overlap(P.data, Q.data)

                if alpha:
                    # penalty term for high power control signals u
                    du += -2 * alpha * u[r, j, k] * dt

                u[r + 1, j, k] = u[r, j, k] + eps * du.real

                if u_limits:
                    if u_limits[0] < u[r + 1, j, k]:
                        u[r + 1, j, k] = u_limits[0]

                    elif u_limits[1] > u[r + 1, j, k]:
                        u[r + 1, j, k] = u_limits[1]

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
    
    return U_f_list[-1], H_td_func, u

def grape_unitary_adaptive(U, H0, H_ops, R, times, eps=None, u_start=None,
                  u_limits=None, interp_kind='linear', use_interp=False,
                  alpha=None, phase_sensitive=True, overlap_terminate=1.0, debug=False,
                  progress_bar=BaseProgressBar()):
    """
    Calculate control pulses for the Hamiltonian operators in H_ops so that the
    unitary U is realized.

    Experimental: Work in progress.
    """

    if eps is None:
        eps = 0.1 * (2 * np.pi) / (times[-1])

    eps_vec = np.array([eps / 2, eps, 2 * eps])
    #eps_vec = [eps]
    best_k = 0
    _k_overlap = np.array([0.0, 0.0, 0.0])

    
    M = len(times)
    J = len(H_ops)
    K = len(eps_vec)
    Uf = [None for _ in range(K)]
    
    u = np.zeros((R, J, M, K))        

    if u_limits and len(u_limits) != 2:
        raise ValueError("u_limits must be a list with two values")

    if u_limits and u_start:
        # make sure that no values in u0 violates the u_limits conditions
        u_start = np.array(u_start)
        u_start[u_start < u_limits[0]] = u_limits[0]
        u_start[u_start > u_limits[1]] = u_limits[1]

    if u_start is not None:
        for idx, u0 in enumerate(u_start):
            for k in range(K):
                u[0, idx, :, k] = u0

    best_k = 1
    _r = 0
    _prev_overlap = 0
    
    progress_bar.start(R)
    for r in range(R - 1):
        progress_bar.update(r)

        _r = r

        if debug:
            print("="  * 80)
            print("eps_vec: ", eps_vec)
        #best_k = 1
        
        _t0 = time.time()

        dt = times[1] - times[0]
        
        if use_interp:
            ip_funcs = [interp1d(times, u[r, j, :, best_k], kind=interp_kind,
                                 bounds_error=False, fill_value=u[r, j, -1, best_k])
                        for j in range(J)]
            def _H_t(t, args=None):
                return H0 + sum([float(ip_funcs[j](t)) * H_ops[j]
                                 for j in range(J)])    

            U_list = [(-1j * _H_t(times[idx]) * dt).expm() for idx in range(M-1)]

        else:
            def _H_idx(idx):
                return H0 + sum([u[r, j, idx, best_k] * H_ops[j] for j in range(J)])    

            U_list = [(-1j * _H_idx(idx) * dt).expm() for idx in range(M-1)]

        if debug:
            print("Time 1: %fs" % (time.time() - _t0))
            _t0 = time.time()
        
        U_f_list = []
        U_b_list = []

        U_f = 1
        U_b = 1
        for m in range(M - 1):

            U_f = U_list[m] * U_f
            U_f_list.append(U_f)

            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - m].dag() * U_b

        if debug:
            print("Time 2: %fs" % (time.time() - _t0))
            _t0 = time.time()

        
        for j in range(J):
            for m in range(M-1):
                P = U_b_list[m] * U
                Q = 1j * dt * H_ops[j] * U_f_list[m]

                #if phase_sensitive:
                #    du = - 2 * _overlap(P, Q) * _overlap(H_ops[j], P) 
                #else:
                du = - cy_overlap(P.data, Q.data)

                if alpha:
                    # penalty term for high power control signals u
                    du += -2 * alpha * u[r, j, m, best_k] * dt

                #print("du", du)

                for k, eps_val in enumerate(eps_vec):
                    u[r + 1, j, m, k] = u[r, j, m, k] + eps_val * du.real
                    
                    if u_limits:
                        if u_limits[0] < u[r + 1, j, m, k]:
                            u[r + 1, j, m, k] = u_limits[0]
                        elif u_limits[1] > u[r + 1, j, m, k]:
                            u[r + 1, j, m, k] = u_limits[1]

            u[r + 1, j, -1, :] = u[r + 1, j, -2, :]


        if debug:
            print("Time 3: %fs" % (time.time() - _t0))
            _t0 = time.time()

        for k, eps_val in enumerate(eps_vec):
                             
            def _H_idx(idx):
                return H0 + sum([u[r + 1, j, idx, k] * H_ops[j] for j in range(J)])    

            U_list = [(-1j * _H_idx(idx) * dt).expm() for idx in range(M-1)]
            
            Uf[k] = gate_sequence_product(U_list)
            _k_overlap[k] = cy_overlap(Uf[k].data, U.data).real

        best_k = np.argmax(_k_overlap)
        if debug:
            print("k_overlap: ", _k_overlap, best_k)
            
        if _prev_overlap > _k_overlap[best_k]:
            if debug:
                print("Regression, stepping back with smaller eps.")
                
            u[r + 1, :, :, :] = u[r, :, :, :]
            eps_vec /= 2
        else:
                
            if best_k == 0:
                eps_vec /= 2
            
            elif best_k == 2:
                eps_vec *= 2
        
            _prev_overlap = _k_overlap[best_k]
            
        if overlap_terminate < 1.0:
            if _k_overlap[best_k] > overlap_terminate:
                print("Reached target fidelity, terminating.")
                break

        if debug:
            print("Time 4: %fs" % (time.time() - _t0))
            _t0 = time.time()
                
    if use_interp:
        ip_funcs = [interp1d(times, u[_r, j, :, best_k], kind=interp_kind,
                             bounds_error=False, fill_value=u[R - 1, j, -1])
                    for j in range(J)]

        H_td_func = [H0] + [[H_ops[j], lambda t, args, j=j: ip_funcs[j](t)]
                              for j in range(J)]    
    else:
        H_td_func = [H0] + [[H_ops[j], u[_r, j, :, best_k]] for j in range(J)]    

    progress_bar.finished()
    
    return Uf[best_k], H_td_func, u[:_r,:,:,best_k]

