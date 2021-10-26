# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR,
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
import numpy as np
import qutip.settings as qset


def check_use_openmp(options):
    """
    Check to see if OPENMP should be used in dynamic solvers.
    """
    force_omp = False
    if qset.has_openmp and options.use_openmp is None:
        options.use_openmp = True
        force_omp = False
    elif qset.has_openmp and options.use_openmp == True:
        force_omp = True
    elif qset.has_openmp and options.use_openmp == False:
        force_omp = False
    elif qset.has_openmp == False and options.use_openmp == True:
        raise Exception('OPENMP not available.')
    else:
        options.use_openmp = False
        force_omp = False
    #Disable OPENMP in parallel mode unless explicitly set.    
    if not force_omp and os.environ['QUTIP_IN_PARALLEL'] == 'TRUE':
        options.use_openmp = False


def use_openmp():
    """
    Check for using openmp in general cases outside of dynamics
    """
    if qset.has_openmp and os.environ['QUTIP_IN_PARALLEL'] != 'TRUE':
        return True
    else:
        return False


def openmp_components(ptr_list):
    return np.array([ptr[-1] >= qset.openmp_thresh for ptr in ptr_list], dtype=bool)
    