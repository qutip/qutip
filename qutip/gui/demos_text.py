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
###########################################################################
from scipy import arange
#basic demos
basic_labels=["Schrodingers Cat","Q-function","Qobj Eigenvalues/Eigenvectors","blank","blank","blank","blank"]

basic_desc=['Schrodinger Cat state formed from a superposition of \ntwo coherent states.',
                            "Q-function from superposition of two coherent states.",
                            "Eigenvalues/Eigenvectors of cavity-qubit system \nin strong-coupling regime.",
                            "Bloch Sphere","blank","blank","blank"]

basic_nums=arange(1,len(basic_labels)+1) #does not start at zero so commandline output numbers match (0=quit in commandline)

#master equation demos
master_labels=["blank","blank","blank","blank","blank","blank","blank"]
master_desc=["blank","blank","blank","blank","blank","blank","blank"]
master_nums=10+arange(len(master_labels))

monte_labels=["blank","blank","blank","blank","blank","blank","blank"]
monte_desc=["blank","blank","blank","blank","blank","blank","blank"]
monte_nums=20+arange(len(monte_labels))

redfield_labels=["blank","blank","blank","blank","blank","blank","blank"]
redfield_desc=["blank","blank","blank","blank","blank","blank","blank"]
redfield_nums=30+arange(len(redfield_labels))

td_labels=["blank","blank","blank","blank","blank","blank","blank"]
td_desc=["blank","blank","blank","blank","blank","blank","blank"]
td_nums=40+arange(len(td_labels))


#variables to be sent to Examples GUI
tab_labels=['Basic Operations','Master Equation','Monte Carlo','Bloch-Redfield','Time-Dependent']
button_labels=[basic_labels,master_labels,monte_labels,redfield_labels,td_labels]
button_desc=[basic_desc,master_desc,monte_desc,redfield_desc,td_desc]
button_nums=[basic_nums,master_nums,monte_nums,redfield_nums,td_nums]