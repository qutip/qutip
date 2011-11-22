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

import sys
sys.path.append('..')
from qutip import *

import unittest

class TestEvolution(unittest.TestCase):

    """
    A test class for the QuTiP functions for the evolution of quantum systems
    """

    def setUp(self):
        """
        setup
        """
        
    def qubit_integrate(self, tlist, psi0, epsilon, delta, g1, g2, solver):

        H = epsilon / 2.0 * sigmaz() + delta / 2.0 * sigmax()
        
        c_op_list = []

        rate = g1
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sigmam())
    
        rate = g2
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sigmaz())
   
        if solver == "ode":
            expt_list = odesolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
        elif solver == "es":
            expt_list = essolve(H, psi0, tlist, c_op_list, [sigmax(), sigmay(), sigmaz()])  
        elif solver == "mc":
            ntraj = 750
            expt_list = mcsolve(H, psi0, tlist, ntraj, c_op_list, [sigmax(), sigmay(), sigmaz()])  
        else:
            raise ValueError("unknown solver")
        
        return expt_list[0], expt_list[1], expt_list[2]


    def testODESolverCase1(self):
        # with dissipation

        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta   = 1.0 * 2 * pi   # atom frequency
        g2 = 0.1                 # 
        g1 = 0.0                 # 
        psi0 = basis(2,0)        # initial state
        tlist = linspace(0,5,200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "ode")

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
        sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

        self.assertEqual(max(abs(sx - sx_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sy - sy_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sz - sz_analytic)) < 0.05, True)  

    def testODESolverCase2(self):
        # without dissipation

        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta   = 1.0 * 2 * pi   # atom frequency
        g2 = 0.0                 # 
        g1 = 0.0                 # 
        psi0 = basis(2,0)        # initial state
        tlist = linspace(0,5,200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "ode")

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
        sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

        self.assertEqual(max(abs(sx - sx_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sy - sy_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sz - sz_analytic)) < 0.05, True)  

    def testESSolverCase1(self):
        
        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta   = 1.0 * 2 * pi   # atom frequency
        g2 = 0.1                 # 
        g1 = 0.0                 # 
        psi0 = basis(2,0)        # initial state
        tlist = linspace(0,5,200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "es")

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
        sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

        self.assertEqual(max(abs(sx - sx_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sy - sy_analytic)) < 0.05, True)
        self.assertEqual(max(abs(sz - sz_analytic)) < 0.05, True)  

    def testMCSolverCase1(self):
        # with dissipation
        
        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta   = 1.0 * 2 * pi   # atom frequency
        g2 = 0.1                 # 
        g1 = 0.0                 # 
        psi0 = basis(2,0)        # initial state
        tlist = linspace(0,5,200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "mc")

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
        sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

        self.assertEqual(max(abs(sx - sx_analytic)) < 0.25, True)
        self.assertEqual(max(abs(sy - sy_analytic)) < 0.25, True)
        self.assertEqual(max(abs(sz - sz_analytic)) < 0.25, True)  

    def testMCSolverCase2(self):
        # without dissipation

        epsilon = 0.0 * 2 * pi   # cavity frequency
        delta   = 1.0 * 2 * pi   # atom frequency
        g2 = 0.0                 # 
        g1 = 0.0                 # 
        psi0 = basis(2,0)        # initial state
        tlist = linspace(0,5,200)

        sx, sy, sz = self.qubit_integrate(tlist, psi0, epsilon, delta, g1, g2, "mc")

        sx_analytic = zeros(shape(tlist))
        sy_analytic = -sin(2*pi*tlist) * exp(-tlist * g2)
        sz_analytic = cos(2*pi*tlist) * exp(-tlist * g2)

        self.assertEqual(max(abs(sx - sx_analytic)) < 0.25, True)
        self.assertEqual(max(abs(sy - sy_analytic)) < 0.25, True)
        self.assertEqual(max(abs(sz - sz_analytic)) < 0.25, True)  

if __name__ == '__main__':

    unittest.main()
