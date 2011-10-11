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
import sys, string


class Codegen():
    """
    Class for generating cython code files at runtime.
    """
    def __init__(self,tab="\t"):
        self.code=[]
        self.tab=tab
        self.level=0
    #write lines of code to self.code
    def write(self,string):
        self.code.append(self.tab*self.level+string+"\n")
    #open file called filename for writing    
    def file(self,filename="rhs.pyx"):
        self.file=open(filename,"w")
    #generate the file    
    def generate(self):
        self.file.writelines(self.code)
        self.file.close()
    #increase indention level by one
    def indent(self):
        self.level+=1
    #decrease indention level by one
    def dedent(self):
        if self.level==0:
            raise SyntaxError("Error in code generator")
        self.level-=1
    
        
def cython_preamble():
    """
    Returns list for standard Cython file preamble.
    """
    line0="# This file is generated automatically by QuTiP. (c)2011 Paul D. Nation & J. R. Johansson"
    line1="import numpy as np"
    line2="cimport numpy as np"
    line3="cimport cython"
    line4=""
    line5="ctypedef np.complex128_t CTYPE_t"
    line6="ctypedef np.float64_t DTYPE_t"
    return [line0,line1,line2,line3,line4,line5,line6]



if __name__=="__main__":
    cgen=Codegen()
    cgen.file()
    for line in cython_preamble():
        cgen.write(line)
    cgen.generate()
