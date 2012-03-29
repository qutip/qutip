#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################


from qutip import *

import unittest

class TestFileIO(unittest.TestCase):

    """
    A test class for the QuTiP functions for writing and reading data to files.
    """

    def setUp(self):
        """
        setup
        """

    def testRWRealDefault(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N))

        file_data_store("test.dat", data)
        data2 = file_data_read("test.dat")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

    def testRWRealDecimal(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N))

        file_data_store("test.dat", data, "real", "decimal")
        data2 = file_data_read("test.dat", ",")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

    def testRWRealExp(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N))

        file_data_store("test.dat", data, "real", "exp")
        data2 = file_data_read("test.dat", ",")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

    def testRWComplexDefault(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N)) + 1j*(1-2*rand(N,N))

        file_data_store("test.dat", data)
        data2 = file_data_read("test.dat")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

    def testRWComplexDecimal(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N)) + 1j*(1-2*rand(N,N))

        file_data_store("test.dat", data, "complex", "decimal")
        data2 = file_data_read("test.dat", ",")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

    def testRWComplexExp(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N)) + 1j*(1-2*rand(N,N))

        file_data_store("test.dat", data, "complex", "exp")
        data2 = file_data_read("test.dat", ",")

        # make sure the deviation is small:
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 


    def testRWSeparatorDetection(self):

        # create some random data
        N = 10
        data = (1-2*rand(N,N)) + 1j*(1-2*rand(N,N))

        # comma separated values
        file_data_store("test.dat", data, "complex", "exp", ",")
        data2 = file_data_read("test.dat")
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

        # semicolon separated values
        file_data_store("test.dat", data, "complex", "exp", ";")
        data2 = file_data_read("test.dat")
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

        # tab separated values
        file_data_store("test.dat", data, "complex", "exp", "\t")
        data2 = file_data_read("test.dat")
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

        # space separated values
        file_data_store("test.dat", data, "complex", "exp", " ")
        data2 = file_data_read("test.dat")
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 

        # mixed-whitespace separated values
        file_data_store("test.dat", data, "complex", "exp", " \t ")
        data2 = file_data_read("test.dat")
        self.assertTrue(amax(abs((data-data2))) < 1e-8) 
        


if __name__ == '__main__':

    unittest.main()
