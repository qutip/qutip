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
from qstate import qstate
from Qobj import dag

# FREDKIN computes the operator for a Fredkin gate
# The last two inputs are swapped if the first input is u

# A B C  A' B' C'
# ---------------
# d d d  d  d  d
# d d u  d  d  u
# d u d  d  u  d
# d u u  d  u  u
# u d d  u  d  d
# u d u  u  u  d
# u u d  u  d  u
# u u u  u  u  u


def fredkin():
	uuu = qstate('uuu')
	uud = qstate('uud') 
	udu = qstate('udu')
	udd = qstate('udd')
	duu = qstate('duu') 
	dud = qstate('dud')
	ddu = qstate('ddu')
	ddd = qstate('ddd')
	Q = ddd*dag(ddd) + ddu*dag(ddu) + dud*dag(dud) + duu*dag(duu) + udd*dag(udd) + uud*dag(udu) + udu*dag(uud) + uuu*dag(uuu)
	return Q

