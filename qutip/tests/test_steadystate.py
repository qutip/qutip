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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################


from qutip import *
from numpy.testing import assert_equal


def test_qubit():
    "Steady state: Thermal qubit"
    # thermal steadystate of a qubit: compare numerics with analytical formula
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H = 0.5 * 2 * pi * sz
    gamma1 = 0.05   
    
    wth_vec = linspace(0.1,3,20) 
    p_ss = zeros(shape(wth_vec))    

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (exp(1.0/wth)-1) # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(sqrt(rate) * sm)
        rate = gamma1 * n_th
        c_op_list.append(sqrt(rate) * sm.dag())
        rho_ss = steadystate(H, c_op_list)
        p_ss[idx] = expect(sm.dag() * sm, rho_ss)

    p_ss_analytic = exp(-1.0/wth_vec)/(1+exp(-1.0/wth_vec))
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-5,True)

def test_ho():
    "Steady state: Thermal harmonic oscillator"
    # thermal steadystate of an oscillator: compare numerics with analytical formula
    a = destroy(40)
    H = 0.5 * 2 * pi * a.dag() * a
    gamma1 = 0.05   
    
    wth_vec = linspace(0.1,3,20) 
    p_ss = zeros(shape(wth_vec))    

    for idx, wth in enumerate(wth_vec):

        n_th = 1.0 / (exp(1.0/wth)-1) # bath temperature
        c_op_list = []
        rate = gamma1 * (1 + n_th)
        c_op_list.append(sqrt(rate) * a)
        rate = gamma1 * n_th
        c_op_list.append(sqrt(rate) * a.dag())
        rho_ss = steadystate(H, c_op_list)
        p_ss[idx] = real(expect(a.dag() * a, rho_ss))

    p_ss_analytic = 1.0 / (exp(1.0/wth_vec)-1)
    delta = sum(abs(p_ss_analytic - p_ss))
    assert_equal(delta < 1e-3,True)


