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
2016-03-9: This module has been deprecated.
The code has been transfered (and transformed) into:
qutip.nonmarkov.hsolver
It is now a class type solver.
"""

# Author: Neill Lambert, Anubhav Vardhan
# Contact: nwlambert@gmail.com

__all__ = ['hsolve']

import warnings
from qutip.nonmarkov.hsolver import HSolverDL
warnings.simplefilter('always', DeprecationWarning) #turn off filter

def hsolve(H, psi0, tlist, Q, gam, lam0, Nc, N, w_th, options=None):
    """
    Function to solve for an open quantum system using the
    hierarchy model.

    Parameters
    ----------
    H: Qobj
        The system hamiltonian.
    psi0: Qobj
        Initial state of the system.
    tlist: List.
        Time over which system evolves.
    Q: Qobj
        The coupling between system and bath.
    gam: Float
        Bath cutoff frequency.
    lam0: Float
        Coupling strength.
    Nc: Integer
        Cutoff parameter.
    N: Integer
        Number of matsubara terms.
    w_th: Float
        Temperature.
    options : :class:`qutip.Options`
        With options for the solver.

    Returns
    -------
    output: Result
        System evolution.
    """

    warnings.warn("This function has been deprecated. "
        "You should switch to using the run method of "
        "nonmarkov.hsolver.HSolverDL",
        DeprecationWarning)

    hsolver = HSolverDL(H, Q, lam0, w_th, Nc, N, gam,
                         renorm=True, bnd_cut_approx=True,
                         options=options, stats=True)

    output = hsolver.run(psi0, tlist)

    return output
