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
import os
import numpy
from scipy import ndarray, array

from qutip.cyQ.codegen import Codegen
from qutip.odeoptions import Odeoptions
from qutip.odechecks import _ode_checks
from qutip.odeconfig import odeconfig
from qutip.qobj import Qobj
from qutip.superoperator import spre, spost


def rhs_clear():
    """
    Resets the string-format time-dependent Hamiltonian parameters.

    Parameters
    ----------

    Returns
    -------
    Nothing, just clears data from internal odeconfig module.

    """
    # time-dependent (TD) function stuff
    odeconfig.tdfunc = None     # Placeholder for TD RHS function.
    odeconfig.colspmv = None    # Placeholder for TD col-spmv function.
    odeconfig.colexpect = None  # Placeholder for TD col_expect function.
    odeconfig.string = None     # Holds string of variables to be passed onto
                                # time-depdendent ODE solver.
    odeconfig.tdname = None     # Name of td .pyx file
                                # (used in parallel mc code)


def rhs_generate(H, c_ops, args={}, options=Odeoptions(), name=None):
    """
    Generates the Cython functions needed for solving the dynamics of a
    given system using the mesolve function inside a parfor loop.

    Parameters
    ----------
    H : qobj
        System Hamiltonian.
    c_ops : list
        ``list`` of collapse operators.
    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.
    options : Odeoptions
        Instance of ODE solver options.
    name: str
        Name of generated RHS

    Notes
    -----
    Using this function with any solver other than the mesolve function
    will result in an error.

    """
    odeconfig.reset()
    odeconfig.options = options

    if name:
        odeconfig.tdname = name
    else:
        odeconfig.tdname = "rhs" + str(odeconfig.cgen_num)

    Lconst = 0

    Ldata = []
    Linds = []
    Lptrs = []
    Lcoeff = []

    # loop over all hamiltonian terms, convert to superoperator form and
    # add the data of sparse matrix represenation to
    for h_spec in H:
        if isinstance(h_spec, Qobj):
            h = h_spec
            Lconst += -1j * (spre(h) - spost(h))

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

            L = -1j * (spre(h) - spost(h))

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append(h_coeff)

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected string format)")

    # loop over all collapse operators
    for c_spec in c_ops:
        if isinstance(c_spec, Qobj):
            c = c_spec
            cdc = c.dag() * c
            Lconst += spre(
                c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)

        elif isinstance(c_spec, list):
            c = c_spec[0]
            c_coeff = c_spec[1]

            cdc = c.dag() * c
            L = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) - 0.5 * spost(cdc)

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append("(" + c_coeff + ")**2")

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected string format)")

    # add the constant part of the lagrangian
    if Lconst != 0:
        Ldata.append(Lconst.data.data)
        Linds.append(Lconst.data.indices)
        Lptrs.append(Lconst.data.indptr)
        Lcoeff.append("1.0")

    # the total number of liouvillian terms (hamiltonian terms + collapse
    # operators)
    n_L_terms = len(Ldata)

    cgen = Codegen(h_terms=n_L_terms, h_tdterms=Lcoeff, args=args,
                   odeconfig=odeconfig)
    cgen.generate(odeconfig.tdname + ".pyx")

    code = compile('from ' + odeconfig.tdname +
                   ' import cyq_td_ode_rhs', '<string>', 'exec')
    exec(code, globals())

    odeconfig.tdfunc = cyq_td_ode_rhs
    try:
        os.remove(odeconfig.tdname + ".pyx")
    except:
        pass
