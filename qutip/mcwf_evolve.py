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

from qobj import *
import random

from scipy.integrate import *

#
#
# 
def mcwf_evolve(tlist, H, psi0, ntraj, c_op_list, expt_op_list):
    """!
    @brief Evolve the wave function using the Monte Carlo algorithm.
    Ref: K Molmer et. al, J. Opt. Soc. Am. B, Vol. 10, No. 3, March 1993
    Use loop over monte-carlo iterations as outer loop
    """
    n_tsteps  = len(tlist)
    n_expt_op = len(expt_op_list)
    n_op      = len(c_op_list)
    expt_list = zeros([n_expt_op, n_tsteps])
    dp_list   = zeros([n_op])
    dt        = tlist[1]-tlist[0]

    Heff = H
    cdc_op_list = []
    for c in c_op_list:
        cdc = c.dag()*c
        Heff -= 0.5j * cdc
        cdc_op_list.append(cdc)

    for n in range(0, ntraj):

        print "n =", n, "of", ntraj

        psi = psi0
        for t_idx in range(0, n_tsteps):

            # calculate all the expectation values: contribution from single trajectory
            for m in range(0, n_expt_op):
                expt_list[m,t_idx] += expect(expt_op_list[m], psi) / ntraj

            # calculate wavefunction at psi(t+dt) 
            dp = 0    
            for m in range(0, n_op):
                dp_list[m] = dt * expect(cdc_op_list[m], psi)
                dp += dp_list[m] 

            # randomly select to do a quantum jump or not
            if dp < random.random():
                # no quantum jump
                psi = (psi - 1j * dt * Heff * psi) / sqrt(1 - dp)
            
            else:
                # quantum jump                  
                m = (random.random() < cumsum(dp_list / dp)).tolist().index(True)
                psi = c_op_list[m] * psi / sqrt(dp_list[m] / dp)
           
    return expt_list


# 
#
# Ref: K Molmer et. al, J. Opt. Soc. Am. B, Vol. 10, No. 3, March 1993
#
# XXX: should loop over ntraj or t be the outer loop?
#
# Use loop over time steps as outer loop
#
def mcwf_evolve_slow(tlist, H, psi0, ntraj, c_op_list, expt_op_list):
    """!
    Evolve the wave function using the Monte Carlo method. 
    Use loop over time steps as outer loop
    """
    n_tsteps  = len(tlist)
    n_expt_op = len(expt_op_list)
    n_op      = len(c_op_list)
    expt_list = zeros([n_expt_op, n_tsteps])
    dp_list   = zeros([n_op])
    dt        = tlist[1]-tlist[0]

    Heff = H
    cdc_op_list = []
    for c in c_op_list:
        cdc = c.dag()*c
        Heff -= 0.5j * cdc
        cdc_op_list.append(cdc)

    psi = psi0
    for t_idx in range(0, n_tsteps):

        print "psi(", t_idx, "of", n_tsteps, ") =", psi.norm()

        # calculate all the expectation values
        for m in range(0, n_expt_op):
            expt_list[m,t_idx] = expect(expt_op_list[m], psi)

        # calculate wavefunction at psi(t+dt) ntraj times an take the average
        psi_sum = 0
        for n in range(0, ntraj):

            dp = 0    
            for m in range(0, n_op):
                dp_list[m] = dt * expect(cdc_op_list[m], psi)
                dp += dp_list[m] 

            # randomly select to do a quantum jump or not
            e = random.random()
                
            if dp < e:
                # no quantum jump
                psi_sum += (psi - 1j * dt * Heff * psi) / sqrt(1 - dp)
            
            else:
                # quantum jump                  
                e = random.random() # reuse old e?
                m = (e < cumsum(dp_list / dp)).tolist().index(True)
                psi_sum += c_op_list[m] * psi / sqrt(dp_list[m] / dp)

        #psi = psi_sum / ntraj
        psi = psi_sum / psi_sum.norm()

    return expt_list



#
# 
# Ref: K Molmer et. al, J. Opt. Soc. Am. B, Vol. 10, No. 3, March 1993
# Ref: Quantum trajectories, Milburn and Wiseman
#
#
def mcwf_evolve_alt(tlist, H, psi0, ntraj, c_op_list, expt_op_list):
    """!
    Evolve the wave function 
    """
    n_tsteps  = len(tlist)
    n_expt_op = len(expt_op_list)
    n_op      = len(c_op_list)
    expt_list = zeros([n_expt_op, n_tsteps])
    dp_list   = zeros([n_op])
    dt        = tlist[1]-tlist[0]

    Heff = H
    cdc_op_list = []
    for c in c_op_list:
        cdc = c.dag()*c
        Heff -= 0.5j * cdc
        cdc_op_list.append(cdc)

    for n in range(0, ntraj):

        print "n =", n, "of", ntraj

        psi = psi0
        for t_idx in range(0, n_tsteps):

            # calculate all the expectation values: contribution from single trajectory
            for m in range(0, n_expt_op):
                expt_list[m,t_idx] = expect(expt_op_list[m], psi) / ntraj

            r = random.random()

            # integrate the eqm until <psi(T)|psi(T)> = r
            psiT = mcwf_ode(Heff, psi, r)

            print psiT

            # calculate wavefunction at psi(t+dt) 
            dp = 0    
            for m in range(0, n_op):
                print cdc_op_list[m]
                dp_list[m] = dt * expect(cdc_op_list[m], psiT)
                dp += dp_list[m] 

            print "cumsum =", cumsum(dp_list)

            # quantum jump                  
            m = (random.random() < cumsum(dp_list / dp)).tolist().index(True)
            psi = c_op_list[m] * psiT / sqrt(dp_list[m] / dp)

           
    return expt_list


#
#
#
#def mcwf_ode_func(psi, t, Heff):
def mcwf_ode_func(t, psi, Heff):

    print "---------------- mcwf_ode_func(", t, ") ----------------"
    #print "Heff =", Heff
    print "psi =", type(psi)

    return psi
    
    #mat=sp.csr_matrix([[0,sqrt(t)],[1,0]])
    #ret=mat*y #cannot use dot(a,b) since mat is mtrix and not array
    #return ret.T #to column vec


#
#
#
def mcwf_odeint(Heff, psi, R):
        
    dt = 0.01
    print "psi =", psi
    psiT = psi
    t_vec = linspace(0, 0.1, 5)

#   while psiT.norm() > R:
#    psiT = odeint(mcwf_ode_func, asarray(psiT.full()), t_vec, args=(Heff,))
    psiTdata = odeint(mcwf_ode_func, asarray(psiT.full()).T, t_vec, args=(Heff,))

    psiT = qobj(psiTdata.T)
    print "psiT =", psiT
    return psiT

#
#
#
def mcwf_ode(Heff, psi, R):
        
    dt = 0.01

    #psiT = psi

    #print "psiT =", psiT

    r = scipy.integrate.ode(mcwf_ode_func)

    initial_vector = asarray(psi.full())[:]

    print "initial vector = ", initial_vector

    r.set_initial_value(initial_vector, 0.0).set_f_params(Heff)
    
    while r.successful() and r.t < 0.1:
        r.integrate(r.t + 0.01)
        print "ITER: ", r.t, r.y

    psiT = qobj(r.y)

    print "psiT =", psiT

    return psiT
    





#>>> from scipy import eye
#>>> from scipy.integrate import ode
#>>>
#>>> y0, t0 = [1.0j, 2.0], 0
#>>>
##>>> def f(t, y, arg1):
#>>>     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
#>>> def jac(t, y, arg1):
#>>>     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
#The integration:
#>>> r = ode(f, jac).set_integrator('zvode', method='bdf', with_jacobian=True)
#>>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
#>>> t1 = 10
#>>> dt = 1
#>>> while r.successful() and r.t < t1:
#>>>     r.integrate(r.t+dt)
#>>>     print r.t, r.y


















