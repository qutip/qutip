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
import numpy as np
from qutip import odeconfig
class Codegen():
    """
    Class for generating cython code files at runtime.
    """
    def __init__(self,h_terms=None,h_tdterms=None,h_const=None,c_terms=None,c_tdterms=None,c_const=None,tab="\t"):
        import sys,os
        sys.path.append(os.getcwd())
        
        #--- Hamiltonian time-depdendent pieces ----#
        self.h_terms=h_terms
        self.h_tdterms=h_tdterms
        self.h_const=h_const
        #--- Collapse operator time-depdendent pieces ----#
        self.c_terms=c_terms
        self.c_tdterms=c_tdterms
        self.c_const=c_const
        #--- Code generator properties----#
        self.code=[]
        self.tab=tab
        self.level=0
        self.func_list=[func+'(' for func in dir(np.math)[4:-1]] #add a '(' on the end to guarentee function is selected 
    #write lines of code to self.code
    def write(self,string):
        self.code.append(self.tab*self.level+string+"\n")
    #open file called filename for writing    
    def file(self,filename):
        self.file=open(filename,"w")
    #generate the file    
    def generate(self,filename="rhs.pyx"):
        self.time_vars()
        for line in cython_preamble():
            self.write(line)
        if self.h_terms:
            for line in cython_checks()+self.ODE_func_header():
                self.write(line)
            self.indent()
            for line in self.func_vars():
                self.write(line)
            for line in self.func_for():
                self.write(line)
            self.write(self.func_end())
            self.dedent()
            for line in cython_checks()+cython_spmv():
                 self.write(line)
        
        #generate collapse operator function if any c_terms
        if self.c_terms:
            for line in cython_checks()+self.col_spmv_header()+cython_col_spmv():
                self.write(line)
            self.indent()
            for line in self.func_which():
                self.write(line)
            self.write(self.func_end())
            self.dedent()
        
        self.file(filename)
        self.file.writelines(self.code)
        self.file.close()
        odeconfig.cgen_num+=1
    #increase indention level by one
    def indent(self):
        self.level+=1
    #decrease indention level by one
    def dedent(self):
        if self.level==0:
            raise SyntaxError("Error in code generator")
        self.level-=1
    def ODE_func_header(self):
        """
        Creates function header for time-dependent ODE RHS.
        """
        func_name="def cyq_td_ode_rhs("
        input_vars="float t, np.ndarray[CTYPE_t, ndim=1] vec" #strings for time and vector variables
        for k in xrange(self.h_terms):
            input_vars+=", np.ndarray[CTYPE_t, ndim=1] data"+str(k)+", np.ndarray[int, ndim=1] idx"+str(k)+", np.ndarray[int, ndim=1] ptr"+str(k)
        if self.h_const:
            td_consts=self.h_const.items()
            td_len=len(td_consts)
            for jj in range(td_len):
                kind=type(td_consts[jj][1]).__name__
                input_vars+=", np."+kind+"_t"+" "+td_consts[jj][0]
        func_end="):"
        return [func_name+input_vars+func_end]
    
    def col_spmv_header(self):
        """
        Creates function header for time-dependent
        collapse operator terms.
        """
        func_name="def col_spmv("
        input_vars="int which, float t, np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[CTYPE_t, ndim=2] vec"
        if self.c_const:
            td_consts=self.c_const.items()
            td_len=len(td_consts)
            for jj in range(td_len):
                kind=type(td_consts[jj][1]).__name__
                input_vars+=", np."+kind+"_t"+" "+td_consts[jj][0]
        func_end="):"
        return [func_name+input_vars+func_end]
    def time_vars(self):
        """
        Rewrites time-dependent parts to include np.
        """
        if self.h_terms:
            for jj in xrange(self.h_terms):
                text=self.h_tdterms[jj]
                any_np=np.array([text.find(x) for x in self.func_list])
                ind=np.nonzero(any_np>-1)[0]
                for kk in ind:
                    new_text='np.'+self.func_list[kk]
                    text=text.replace(self.func_list[kk],new_text)
                self.h_tdterms[jj]=text
        if self.c_tdterms:
            for jj in xrange(self.c_terms):
                text=self.c_tdterms[jj]
                any_np=np.array([text.find(x) for x in self.func_list])
                ind=np.nonzero(any_np>-1)[0]
                for kk in ind:
                    new_text='np.'+self.func_list[kk]
                    text=text.replace(self.func_list[kk],new_text)
                self.c_tdterms[jj]=text

    def func_vars(self):
        """
        Writes the variables and their types & spmv parts
        """
        func_vars=["",'cdef Py_ssize_t row','cdef int num_rows = len(vec)','cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)']
        func_vars.append(" ") #add a spacer line between variables and Hamiltonian components.
        terms=self.h_terms
        tdterms=self.h_tdterms
        for ht in xrange(terms):
            hstr=str(ht)
            str_out="cdef np.ndarray[CTYPE_t, ndim=2] Hvec"+hstr+" = "+"spmv(data"+hstr+","+"idx"+hstr+","+"ptr"+hstr+","+"vec"+")"
            str_out+=" * "+tdterms[ht]
            func_vars.append(str_out)
        return func_vars
    def func_for(self):
        """
        Writes function for-loop
        """
        func_terms=["","for row in range(num_rows):"]
        sum_string="\tout[row,0] = Hvec0[row,0]"
        for ht in xrange(1,self.h_terms):
            sum_string+=" + Hvec"+str(ht)+"[row,0]"
        func_terms.append(sum_string)
        return func_terms
    
    def func_which(self):
        """
        Writes 'else-if' statements for
        collapse operator eval fucntion
        """
        out_string=[]
        for k in xrange(self.c_terms):
            out_string.append("if which == "+str(k)+":")
            out_string.append("\tout*= "+self.c_tdterms[k])
        return out_string
    def func_end(self):
        return "return out"
        
#
# Alternative implementation of the Cython code generator. Include the
# parameters directly in the generated code file instead of passing
# them as arguments through the ODE solver callback function
#       
class Codegen2():
    """
    Class for generating cython code files at runtime.
    """
    def __init__(self, n_L_terms, L_coeffs, args, tab="\t"):
        import sys,os
        sys.path.append(os.getcwd())
        self.n_L_terms = n_L_terms
        self.L_coeffs  = L_coeffs
        self.args      = args
        self.code=[]
        self.tab=tab
        self.level=0
        self.func_list=[func+'(' for func in dir(np.math)[4:-1]] #add a '(' on the end to guarentee function is selected 
        
    #
    # write lines of code to self.code
    #
    def write(self,string):
        self.code.append(self.tab*self.level+string+"\n")
    
    #
    # open file called filename for writing    
    #
    def file(self,filename):
        self.file=open(filename,"w")
   
    #
    # generate the file 
    #
    def generate(self,filename="rhs.pyx"):
        self.time_vars()
        for line in cython_preamble()+cython_checks()+self.func_header():
            self.write(line)
        self.indent()
        for line in self.func_vars():
            self.write(line)
        for line in self.func_for():
            self.write(line)
        self.write(self.func_end())
        self.dedent()
        for line in cython_checks()+cython_spmv():
            self.write(line)
        self.file(filename)
        self.file.writelines(self.code)
        self.file.close()
        odeconfig.cgen_num += 1        
        
    #increase indention level by one
    def indent(self):
        self.level+=1
        
    #decrease indention level by one
    def dedent(self):
        if self.level==0:
            raise SyntaxError("Error in code generator")
        self.level-=1
                
    def func_header(self):
        """
        Creates function header for time-dependent ODE RHS.
        """
        func_name  = "def cyq_td_ode_rhs("
        input_vars = "float t, np.ndarray[CTYPE_t, ndim=1] vec, " #strings for time and vector variables
        arg_L_list = []
        for k in range(self.n_L_terms):
            arg_L_list.append("np.ndarray[CTYPE_t, ndim=1] data"+str(k)+", np.ndarray[int, ndim=1] idx"+str(k)+", np.ndarray[int, ndim=1] ptr"+str(k))
        input_vars += ", ".join(arg_L_list)
        func_end="):"
        return [func_name+input_vars+func_end]
        
        
    def time_vars(self):
        """
        Rewrites time-dependent parts to include np.
        """
        for n in range(len(self.L_coeffs)):
            text = self.L_coeffs[n]
            any_np = np.array([text.find(x) for x in self.func_list])
            ind = np.nonzero(any_np>-1)[0]
            for m in ind:
                text = text.replace(self.func_list[m], 'np.'+self.func_list[m])
            self.L_coeffs[n]=text

            
    def func_vars(self):
        """
        Writes the variables and their types & spmv parts
        """
        decl_list = ["",'cdef Py_ssize_t row','cdef int num_rows = len(vec)','cdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)', ""]

        if self.args:
            for name, value in self.args.iteritems():
                kind = type(value).__name__
                decl_list.append("cdef np."+kind+"_t"+" "+name+" = "+str(value))
            decl_list.append("")

        for n in range(self.n_L_terms):
            nstr=str(n)
            str_out="cdef np.ndarray[CTYPE_t, ndim=2] Lvec"+nstr+" = "+"spmv(data"+nstr+","+"idx"+nstr+","+"ptr"+nstr+","+"vec"+") * ("+self.L_coeffs[n]+")"
            decl_list.append(str_out)
            
        return decl_list

    def func_for(self):
        """
        Writes function for-loop
        """
        func_terms=["", "for row in range(num_rows):"]
        sum_str_list = []
        for n in range(self.n_L_terms):
            sum_str_list.append("Lvec"+str(n)+"[row,0]")
        func_terms.append("\tout[row,0] = " + " + ".join(sum_str_list))
        func_terms.append("")
        return func_terms
        
    def func_end(self):
        return "return out"        
        
        
def cython_preamble():
    """
    Returns list of strings for standard Cython file preamble.
    """
    line0="# This file is generated automatically by QuTiP. (C) 2011-2012 Paul D. Nation & J. R. Johansson"
    line1="import numpy as np"
    line2="cimport numpy as np"
    line3="cimport cython"
    line4=""
    line5="ctypedef np.complex128_t CTYPE_t"
    line6="ctypedef np.float64_t DTYPE_t"
    return [line0,line1,line2,line3,line4,line5,line6]

def cython_checks():
    """
    List of strings that turn off Cython checks.
    """
    line0=""
    line1=""
    line2="@cython.boundscheck(False)"
    line3="@cython.wraparound(False)"
    return [line0,line1,line2,line3]


def cython_spmv():
    """
    Writes SPMV function.
    """
    line0="def spmv(np.ndarray[CTYPE_t, ndim=1] data, np.ndarray[int] idx,np.ndarray[int] ptr,np.ndarray[CTYPE_t, ndim=1] vec):"
    line1="\tcdef Py_ssize_t row"
    line2="\tcdef int jj,row_start,row_end"
    line3="\tcdef int num_rows=len(vec)"
    line4="\tcdef complex dot"
    line5="\tcdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)"
    line6="\tfor row in range(num_rows):"
    line7="\t\tdot=0.0"
    line8="\t\trow_start = ptr[row]"
    line9="\t\trow_end = ptr[row+1]"
    lineA="\t\tfor jj in range(row_start,row_end):"
    lineB="\t\t\tdot=dot+data[jj]*vec[idx[jj]]"
    lineC="\t\tout[row,0]=dot"
    lineD="\treturn out"
    return [line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,lineA,lineB,lineC,lineD]

def cython_col_spmv():
    """
    Writes col_SPMV header.
    """
    line1="\tcdef Py_ssize_t row"
    line2="\tcdef int jj,row_start,row_end"
    line3="\tcdef int num_rows=len(vec)"
    line4="\tcdef complex dot"
    line5="\tcdef np.ndarray[CTYPE_t, ndim=2] out = np.zeros((num_rows,1),dtype=np.complex)"
    line6="\tfor row in range(num_rows):"
    line7="\t\tdot=0.0"
    line8="\t\trow_start = ptr[row]"
    line9="\t\trow_end = ptr[row+1]"
    lineA="\t\tfor jj in range(row_start,row_end):"
    lineB="\t\t\tdot=dot+data[jj]*vec[idx[jj],0]"
    lineC="\t\tout[row,0]=dot"
    return [line1,line2,line3,line4,line5,line6,line7,line8,line9,lineA,lineB,lineC]

