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

import numpy as np
from scipy.interpolate import interp1d

from qutip.ui.progressbar import BaseProgressBar, TextProgressBar

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


def grape_unitary(U, H0, H_ops, R, times, eps=None, u_start=None,
                  interp_kind='linear', progress_bar=BaseProgressBar()):
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

    if u_start is not None:
        for idx, u0 in enumerate(u_start):
            u[0, idx, :] = u0

    progress_bar.start(R)
    for r in range(R - 1):
        progress_bar.update(r)
        
        ip_funcs = [interp1d(times, u[r, j, :], kind=interp_kind,
                             bounds_error=False, fill_value=u[r, j, -1])
                    for j in range(J)]

        def H_func(t, args=None):
            return H0 + sum([float(ip_funcs[j](t)) * H_ops[j]
                             for j in range(J)])    

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
            for k in range(M-1):
                du = _overlap(U_b_list[k] * U,
                              1j * dt * H_ops[j] * U_f_list[k])
                u[r + 1, j, k] = u[r, j, k] - eps * du.real

            u[r + 1, j, -1] = u[r + 1, j, -2]
            
    ip_funcs = [interp1d(times, u[R - 1, j, :], kind=interp_kind,
                         bounds_error=False, fill_value=u[R - 1, j, -1])
                for j in range(J)]

    H_list_func = [H0] + [[H_ops[j], lambda t, args, j=j: ip_funcs[j](t)]
                          for j in range(J)]    

    progress_bar.finished()
    
    return U_f_list[-1], H_list_func, u
