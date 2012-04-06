################################################################################
#This file is part of QuTiP.
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
################################################################################
from scipy import arange

#basic demos
basic_labels=["Schrodinger Cat","Q-function","Squeezed State","Ground State","Density Matrix Metrics",
            "Coupled Qubit Energies","Bell State","Cavity-Qubit Steadystate","Binary Entropy","3-Qubit GHZ State"]

basic_desc=['Schrodinger Cat state formed from a superposition of two coherent states.',
            "Calculates the Q-function of Schrodinger cat state formed from two coherent states.",
            "3D Wigner and Q-functions for a squeezed coherent state.",
            "Groundstate properties of an ultra-strongly coupled atom-cavity system.",
            "Show relationship between fidelity and trace distance for pure state denisty matrices.",
            "Calculate Eigenenergies of a coupled three qubit system.",
            "Creation and manipulation of a Bell state density matrix.",
            "Steady state of cavity-qubit system as cavity driving frequency is varied.",
            "Entropy of binary system as probability of being in the excited state is varied.",
            "Plot the density matrix for the 3-qubit GHZ state."]

basic_nums=10+arange(len(basic_labels)) #does not start at zero so commandline output numbers match (0=quit in commandline)

#master equation demos
master_labels=["i-Swap Gate","Vacuum Rabi oscillations","Single-atom lasing","Wigner distribution","Heisenberg spin chain","Steady state","blank"]
master_desc=["Dissipative i-Swap Gate vs. ideal gate. Accuracy of dissipative gate given by fidelity.",
            "Vacuum Rabi oscillations in the Jaynes-Cummings model with dissipation.",
            "Single-atom lasing in a Jaynes-Cummings-like system.",
            "Wigner distributions from the evolution of the Jaynes-Cummings model.",
            "The dynamics of a Heisenberg spin-1/2 chain.",
            "Steady state calculation for a sideband-cooled nanomechanical resonator",
            "blank"]
master_nums=20+arange(len(master_labels))


#monte carlo equation demos
monte_labels=["Mc Cavity+Qubit","blank","MC Ensemble Avg.","blank","blank","blank","blank"]
monte_desc=["Monte Carlo evoution of a coherently driven cavity with a two-level atom.",
            "blank",
            "Ensemble averaging of MC trajectories to master equation for Fock state decay. (be patient)",
            "blank",
            "blank",
            "blank",
            "blank"]
monte_nums=30+arange(len(monte_labels))

#time-dependence examples
td_labels=["Rabi oscillations","Single photon source","blank","blank","blank","blank","blank"]
td_desc=["Rabi oscillations of an atom subject to a time-dependent classical driving field.",
         "Single photon source based on a three level atom strongly coupled to a cavity.",
         "blank","blank","blank","blank","blank"]
td_nums=40+arange(len(td_labels))

# advanced equation demos
advanced_labels=["Nonadiabtic evolution","blank","blank","blank","blank","blank"]
advanced_desc=["Nonadiabatic transformation from a decoupled to a coupled spin chain.",
               "blank","blank","blank","blank","blank"]
advanced_nums=50+arange(len(advanced_labels))

#variables to be sent to Examples GUI
tab_labels=['Basics','Master Eq.','Monte Carlo','Time dependent','Advanced']
button_labels=[basic_labels,master_labels,monte_labels,td_labels,advanced_labels]
button_desc=[basic_desc,master_desc,monte_desc,td_desc,advanced_desc]
button_nums=[basic_nums,master_nums,monte_nums,td_nums,advanced_nums]


qutip_keywords=['basis','Bloch','brmesolve','concurrence','create','destroy','displace',
                'entropy_linear','entropy_mutual','entropy_vn','expect','fidelity','ket2dm',
                'parfor','qeye','qfunc','Qobj','rand_dm','rand_herm','rand_ket','rand_unitary',
                'squeez','tensor','tracedist','wigner']





