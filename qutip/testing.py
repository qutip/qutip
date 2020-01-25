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
from qutip.about import about
from qutip import settings as qset

def run(full=False):
    """
    Run the test scripts for QuTiP.

    Parameters
    ----------
    full: bool
        If True run all test (30 min). Otherwise skip few variants of the
        slowest tests.
    """
    # Call about to get all version info printed with tests
    about()
    import pytest
    real_num_cpu = qset.num_cpus
    real_thresh = qset.openmp_thresh
    if qset.has_openmp:
        # For travis which VMs have only 1 cpu.
        # Make sure the openmp version of the functions are tested.
        qset.num_cpus = 2
        qset.openmp_thresh = 100

    test_options = ["--verbosity=1", "--disable-pytest-warnings", "--pyargs"]
    if not full:
        test_options += ['-m', 'not slow']
    pytest.main(test_options + ["qutip"])
    # runs tests in qutip.tests module only

    # Restore previous settings
    if qset.has_openmp:
        qset.num_cpus = real_num_cpu
        qset.openmp_thresh = real_thresh
