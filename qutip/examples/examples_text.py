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
            "Coupled Qubit Energies","Bell State","Cavity-Qubit Steadystate","Binary Entropy","blank"]

basic_desc=['Schrodinger Cat state formed from a \nsuperposition of two coherent states.',
            "Calculates the Q-function of Schrodinger cat\nstate formed from two coherent states.",
            "3D Wigner and Q-functions for a squeezed \ncoherent state.",
            "Groundstate properties of an ultra-strongly\ncoupled atom-cavity system.",
            "Show relationship between fidelity and trace\ndistance for pure state denisty matrices.",
            "Calculate Eigenenergies of a coupled three qubit\nsystem.",
            "Creation and manipulation of a Bell state density\nmatrix.",
            "Steady state of cavity-qubit system using parfor\nfunction to run in parallel.",
            "Entropy of binary system as probability of being\nin the excited state is varied.",
            "blank"]

basic_nums=10+arange(len(basic_labels)) #does not start at zero so commandline output numbers match (0=quit in commandline)

#master equation demos
master_labels=["i-Swap Gate","blank","blank","blank","blank","blank","blank"]
master_desc=["Dissipative i-Swap Gate vs. ideal gate.  Accuracy\nof dissipative gate given by fidelity.",
            "blank","blank","blank","blank","blank","blank"]
master_nums=20+arange(len(master_labels))


#monte carlo equation demos
monte_labels=["Mc Cavity+Qubit","blank","MC Ensemble Avg.","blank","blank","blank","blank"]
monte_desc=["Monte Carlo evoution of a coherently driven\ncavity with a two-level atom",
            "blank",
            "Ensemble averaging of MC trajectories to master\nequation for Fock state decay. (be patient)",
            "blank",
            "blank",
            "blank",
            "blank"]
monte_nums=30+arange(len(monte_labels))

#bloch-redfield equation demos
redfield_labels=["blank","blank","blank","blank","blank","blank"]
redfield_desc=["blank","blank","blank","blank","blank","blank"]
redfield_nums=40+arange(len(redfield_labels))

#time-dependence examples
td_labels=["blank","blank","blank","blank","blank","blank","blank"]
td_desc=["blank","blank","blank","blank","blank","blank","blank"]
td_nums=50+arange(len(td_labels))

#variables to be sent to Examples GUI
tab_labels=['Basics','Master Eq.','Monte Carlo','Bloch-Redfield','Time-Dependent']
button_labels=[basic_labels,master_labels,monte_labels,redfield_labels,td_labels]
button_desc=[basic_desc,master_desc,monte_desc,redfield_desc,td_desc]
button_nums=[basic_nums,master_nums,monte_nums,redfield_nums,td_nums]


qutip_keywords=['basis','Bloch','brmesolve','concurrence','create','destroy','displace',
                'entropy_linear','entropy_mutual','entropy_vn','expect','fidelity','ket2dm',
                'parfor','qeye','qfunc','Qobj','rand_dm','rand_herm','rand_ket','rand_unitary',
                'squeez','tensor','tracedist','wigner']





