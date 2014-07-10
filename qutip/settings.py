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
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
# QuTiP Graphics (set at qutip import)
qutip_graphics = None
# use auto tidyup
auto_tidyup = True
# detect hermiticity
auto_herm = True
# general absolute tolerance
atol = 1e-12
# use auto tidyup absolute tolerance
auto_tidyup_atol = 1e-12
# number of cpus (set at qutip import)
num_cpus = 1
# flag indicating if fortran module is installed
fortran = False
# debug mode for development
debug = False


def reset():
    from qutip._reset import _reset
    _reset()


def load_rc_file(rc_file):
    """
    Load settings for the qutip RC file, by default .qutiprc in the user's home
    directory.
    """
    global qutip_graphics, auto_tidyup, auto_herm, auto_tidyup_atol, \
        num_cpus, debug, atol

    with open(rc_file) as f:
        for line in f.readlines():
            if line[0] != "#":
                var, val = line.strip().split("=")

                if var == "qutip_graphics":
                    qutip_graphics = "NO" if val == "NO" else "YES"

                elif var == "auto_tidyup":
                    auto_tidyup = True if val == "True" else False

                elif var == "auto_tidyup_atol":
                    auto_tidyup_atol = float(val)

                elif var == "atol":
                    atol = float(val)

                elif var == "auto_herm":
                    auto_herm = True if val == "True" else False

                elif var == "num_cpus":
                    num_cpus = int(val)

                elif var == "debug":
                    debug = True if val == "True" else False
