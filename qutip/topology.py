# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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

__all__ = ['berry_curvature']

from qutip import (Qobj, tensor, basis, qeye, isherm, sigmax, sigmay, sigmaz)
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass


def berry_curvature(eigfs):
    """Computes the discretized Berry curvature on the two dimensional grid
    of parameters. The function works well for cases with no band mixing."""
    nparam0 = eigfs.shape[0]
    nparam1 = eigfs.shape[1]
    nocc = eigfs.shape[2]
    
    b_curv=np.zeros((nparam0-1, nparam1-1), dtype=float)


    for i in range(nparam0-1):
        for j in range(nparam1-1):
            rect_prd = np.identity(nocc, dtype=complex)
            innP0 = np.zeros([nocc, nocc], dtype=complex)
            innP1 = np.zeros([nocc, nocc], dtype=complex)
            innP2 = np.zeros([nocc, nocc], dtype=complex)
            innP3 = np.zeros([nocc, nocc], dtype=complex)

            for k in range(nocc):
                for l in range(nocc):
                    wf0 = eigfs[i, j, k, :]
                    wf1 = eigfs[i+1, j, l, :]
                    innP0[k, l] = np.dot(wf0.conjugate(), wf1)                    
                    
                    wf1 = eigfs[i+1, j, k,:]
                    wf2 = eigfs[i+1, j+1, l,:]
                    innP1[k, l] = np.dot(wf1.conjugate(), wf2)                    

                    wf2 = eigfs[i+1, j+1, k,:]
                    wf3 = eigfs[i,j+1, l,:]
                    innP2[k, l] = np.dot(wf2.conjugate(), wf3)                    
                    
                    wf3 = eigfs[i, j+1, k,:]
                    wf0 = eigfs[i, j, l, :]
                    innP3[k, l] = np.dot(wf3.conjugate(), wf0)                    

            rect_prd = np.dot(rect_prd, innP0)
            rect_prd = np.dot(rect_prd, innP1)
            rect_prd = np.dot(rect_prd, innP2)
            rect_prd = np.dot(rect_prd, innP3)

            dett = np.linalg.det(rect_prd)
            curl_z = np.angle(dett)
            b_curv[i,j] = curl_z

    # also plot Berry phase on each small plaquette of the mesh
#    plaq=w_square.berry_flux([0],individual_phases=True)
    #
    fig, ax = plt.subplots()
#    ax.imshow(plaq.T,origin="lower",
#          extent=(0,2*np.pi,2*np.pi,2*np.pi,))
    ax.imshow(b_curv, origin="lower")    
    ax.set_title("Berry curvature near Dirac cone")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    fig.tight_layout()
    fig.savefig("cone_phases.pdf")


    return b_curv

