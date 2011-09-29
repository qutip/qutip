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
import sys,dis

def varargout():
    """
    @brief Returns the number of outputs requested by the user.
        Acts like Matlab varargout function. 
    
    @return int Number of outputs requested by user
    """
    f = sys._getframe().f_back.f_back
    i = f.f_lasti + 3
    bytecode = f.f_code.co_code
    instruction = ord(bytecode[i])
    while True:
        if instruction == dis.opmap['DUP_TOP']:
            if ord(bytecode[i + 1]) == dis.opmap['UNPACK_SEQUENCE']:
                return ord(bytecode[i + 2])
            i += 4
            instruction = ord(bytecode[i])
            continue
        if instruction == dis.opmap['STORE_NAME']:
            return 1
        if instruction == dis.opmap['UNPACK_SEQUENCE']:
            return ord(bytecode[i + 1])
        return 0



            
