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

class TestJCModelEvolution(unittest.TestCase):

    """
    A test class for the QuTiP functions for the evolution of JC model
    """

    def setUp(self):
        """
        setup
        """

    def jc_steadystate(self, N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist):

        # Hamiltonian
        a  = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))
    
        if use_rwa: 
            # use the rotating wave approxiation
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
               
        # collapse operators
        c_op_list = []
    
        n_th_a = 0.0 # zero temperature
    
        rate = kappa * (1 + n_th_a)
        #if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)
    
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a.dag())
    
        rate = gamma
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm)
    
        rate = pump
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm.dag())


        # find the steady state
        rho_ss = steadystate(H, c_op_list)

        return expect(a.dag() * a, rho_ss), expect(sm.dag() * sm, rho_ss)

    def jc_integrate(self, N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, solver):

        # Hamiltonian
        a  = tensor(destroy(N), qeye(2))
        sm = tensor(qeye(N), destroy(2))
    
        if use_rwa: 
            # use the rotating wave approxiation
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
               
        # collapse operators
        c_op_list = []
    
        n_th_a = 0.0 # zero temperature
    
        rate = kappa * (1 + n_th_a)
        #if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)
    
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * a.dag())
    
        rate = gamma
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm)
    
        rate = pump
        if rate > 0.0:
            c_op_list.append(sqrt(rate) * sm.dag())
    
    
        # evolve and calculate expectation values
        if solver == "mc":
            expt_list = mcsolve(H, psi0, tlist, 250, c_op_list, [a.dag() * a, sm.dag() * sm])  
        if solver == "es":
            expt_list = essolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  
        if solver == "ode":
            expt_list = odesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])  

        return expt_list[0], expt_list[1]
        

    def testCase1(self):
        # test case 1: cavity-qubit interaction, no dissipation

        for solver in ("ode", "es", "mc"):

            use_rwa = True
            N = 8           # number of cavity fock states
            wc = 2*pi*1.0   # cavity frequency
            wa = 2*pi*1.0   # atom frequency
            g  = 2*pi*0.01  # coupling strength
            kappa = 0.0     # cavity dissipation rate
            gamma = 0.0     # atom dissipation rate
            pump  = 0.0     # atom pump rate

            # intial state
            n = N - 2
            psi0 = tensor(basis(N,n), basis(2,1))    # start with an excited atom and maximum number of photons
            tlist = linspace(0, 1000, 2000)

            nc, na = self.jc_integrate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, solver)

            nc_ex = (n + 0.5 * (1 - cos(2*g*sqrt(n+1)*tlist)))
            na_ex = 0.5 * (1 + cos(2*g*sqrt(n+1)*tlist))

            self.assertEqual(max(abs(nc-nc_ex)) < 0.005, True)
            self.assertEqual(max(abs(na-na_ex)) < 0.005, True)
            

    def testCase2(self):
        # test case 2: no interaction, cavity and qubit decay

        for solver in ("ode", "es"):

            use_rwa = True
            N = 8           # number of cavity fock states
            wc = 2*pi*1.0   # cavity frequency
            wa = 2*pi*1.0   # atom frequency
            g  = 2*pi*0.0   # coupling strength
            kappa = 0.005   # cavity dissipation rate
            gamma = 0.01    # atom dissipation rate
            pump  = 0.0     # atom pump rate

            # intial state
            n = N - 2
            psi0 = tensor(basis(N,n), basis(2,1))    # start with an excited atom and maximum number of photons
            tlist = linspace(0, 1000, 2000)

            nc, na = self.jc_integrate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, solver)

            nc_ex = (n + 0.5 * (1 - cos(2*g*sqrt(n+1)*tlist))) * exp(-kappa * tlist)
            na_ex = 0.5 * (1 + cos(2*g*sqrt(n+1)*tlist)) * exp(-gamma * tlist)

            self.assertEqual(max(abs(nc-nc_ex)) < 0.005, True)
            self.assertEqual(max(abs(na-na_ex)) < 0.005, True)


    def testCase3(self):
        # test case 3: with interaction, cavity and qubit decay

        for solver in ("ode", "es"):

            use_rwa = True
            N = 8           # number of cavity fock states
            wc = 2*pi*1.0   # cavity frequency
            wa = 2*pi*1.0   # atom frequency
            g  = 2*pi*0.1   # coupling strength
            kappa = 0.05    # cavity dissipation rate
            gamma = 0.001   # atom dissipation rate
            pump  = 0.25    # atom pump rate

            # intial state
            n = N - 2
            psi0 = tensor(basis(N,n), basis(2,1))    # start with an excited atom and maximum number of photons
            tlist = linspace(0, 200, 500)

            nc, na = self.jc_integrate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist, solver)

            # we don't have any analytics for this parameters, so 
            # compare with the steady state
            nc_ss, na_ss = self.jc_steadystate(N, wc, wa, g, kappa, gamma, pump, psi0, use_rwa, tlist)

            nc_ss = nc_ss * ones(shape(nc))
            na_ss = na_ss * ones(shape(na))

            self.assertEqual(abs(nc[-1]-nc_ss[-1]) < 0.005, True)
            self.assertEqual(abs(na[-1]-na_ss[-1]) < 0.005, True)


if __name__ == '__main__':

    unittest.main()
