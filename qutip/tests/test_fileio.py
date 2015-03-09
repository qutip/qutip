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
from numpy import amax
from numpy.testing import assert_, run_module_suite
import scipy
from qutip import *
from qutip import file_data_store, file_data_read


class TestFileIO:
    """
    A test class for the QuTiP functions for writing and reading data to files.
    """

    def testRWRealDefault(self):
        "Read and write real valued default formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data)
        data2 = file_data_read("test.dat")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWRealDecimal(self):
        "Read and write real valued decimal formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data, "real", "decimal")
        data2 = file_data_read("test.dat", ",")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWRealExp(self):
        "Read and write real valued exp formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data, "real", "exp")
        data2 = file_data_read("test.dat", ",")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWComplexDefault(self):
        "Read and write complex valued default formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N)) + 1j * (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data)
        data2 = file_data_read("test.dat")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWComplexDecimal(self):
        "Read and write complex valued decimal formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N)) + 1j * (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data, "complex", "decimal")
        data2 = file_data_read("test.dat", ",")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWComplexExp(self):
        "Read and write complex valued exp formatted data"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N)) + 1j * (1 - 2 * scipy.rand(N, N))

        file_data_store("test.dat", data, "complex", "exp")
        data2 = file_data_read("test.dat", ",")
        # make sure the deviation is small:
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")

    def testRWSeparatorDetection(self):
        "Read and write with automatic separator detection"

        # create some random data
        N = 10
        data = (1 - 2 * scipy.rand(N, N)) + 1j * (1 - 2 * scipy.rand(N, N))

        # comma separated values
        file_data_store("test.dat", data, "complex", "exp", ",")
        data2 = file_data_read("test.dat")
        assert_(amax(abs((data - data2))) < 1e-8)

        # semicolon separated values
        file_data_store("test.dat", data, "complex", "exp", ";")
        data2 = file_data_read("test.dat")
        assert_(amax(abs((data - data2))) < 1e-8)

        # tab separated values
        file_data_store("test.dat", data, "complex", "exp", "\t")
        data2 = file_data_read("test.dat")
        assert_(amax(abs((data - data2))) < 1e-8)

        # space separated values
        file_data_store("test.dat", data, "complex", "exp", " ")
        data2 = file_data_read("test.dat")
        assert_(amax(abs((data - data2))) < 1e-8)

        # mixed-whitespace separated values
        file_data_store("test.dat", data, "complex", "exp", " \t ")
        data2 = file_data_read("test.dat")
        assert_(amax(abs((data - data2))) < 1e-8)
        os.remove("test.dat")
    
    def testqsaveqload(self):
        "qsave/qload"
        A = sigmax()
        B = num(5)
        C = coherent_dm(10,1j)
        ops = [A, B, C]
        qsave(ops, 'fileio_check')
        ops2 = qload('fileio_check')
        assert_(ops == ops2)
        try:
            os.remove('fileio_check.qu')
        except:
            pass


if __name__ == "__main__":
    run_module_suite()
