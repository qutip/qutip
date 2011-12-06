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
Hdata=None  # List of sparse matrix data
Hinds=None  # List of sparse matrix indices
Hptrs=None  # List of sparse matrix ptrs
cflag=0     # Flag signaling collapse operators
tflag=0     # Flag signaling time-dependent problem
cgen_num=0  # Number of times codegen function has been called in current Python session.
tdfunc=None # Placeholder for time-dependent RHS function.
string=None