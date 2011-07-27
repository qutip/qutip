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
###########################################################################
from scipy import *
from Qobj import *

class eseries:
    __array_priority__=101
    def __init__(self,q=array([]),s=array([])):
        if (not any(q)) and (not any(s)):
            self.ampl=array([])
            self.rates=array([])
            self.dims=[[1,1]] 
            self.shape=[1,1]
        if any(q) and (not any(s)):
            if isinstance(q,eseries):
                self.ampl=q.ampl
                self.rates=q.rates
                self.dims=q.dims
                self.shape=q.shape
            elif isinstance(q,(ndarray,list)):
                ind=shape(q)
                num=ind[0] #number of elements in q
                #sh=array([Qobj(x).shape for x in range(0,num)])
                sh=array([Qobj(x).shape for x in q])
                if any(sh!=sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                #self.ampl=array([Qobj(x) for x in q])
                self.ampl=array([x for x in q])
                self.rates=zeros(ind)
                self.dims=self.ampl[0].dims
                self.shape=self.ampl[0].shape
            elif isinstance(q,Qobj):
                qo=Qobj(q)
                self.ampl=array([qo])
                self.rates=array([0])
                self.dims=qo.dims
                self.shape=qo.shape
            else:
                self.ampl  = array([q])
                self.rates = array([0])
                self.dims  = [[1, 1]]
                self.shape = [1,1]

        if any(q) and any(s): 
            if isinstance(q,(ndarray,list)):
                ind=shape(q)
                num=ind[0]
                sh=array([Qobj(q[x]).shape for x in range(0,num)])
                if any(sh!=sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl=array([Qobj(q[x]) for x in range(0,num)])
                self.dims=self.ampl[0].dims
                self.shape=self.ampl[0].shape
            else:
                num=1
                self.ampl=array([Qobj(q)])
                self.dims=self.ampl[0].dims
                self.shape=self.ampl[0].shape
            if isinstance(s,(int,complex,float)):
                if num!=1:
                    raise TypeError('Number of rates must match number of members in object array.')
                self.rates=array([s])
            elif isinstance(s,(ndarray,list)):
                if len(s)!=num:
                    raise TypeError('Number of rates must match number of members in object array.')
                self.rates=array(s)
        if len(self.ampl)!=0:
            zipped=zip(self.rates,self.ampl)#combine arrays so that they can be sorted together
            zipped.sort() #sort rates from lowest to highest
            rates,ampl=zip(*zipped) #get back rates and ampl
            self.ampl=array(ampl)
            self.rates=array(rates)
    
    ######___END_INIT___######################

    ##########################################            
    def __str__(self):#string of ESERIES information
        self.tidy()
        print "ESERIES object: "+str(len(self.ampl))+" terms"
        print "Hilbert space dimensions: "+str(self.dims)
        for k in range(0,len(self.ampl)):
            print "Exponent #"+str(k)+" = "+str(self.rates[k])
            if isinstance(self.ampl[k], sp.spmatrix):
                print self.ampl[k].full()
            else:
                print self.ampl[k]
        return ""
    def __add__(self,other):#Addition with ESERIES on left (ex. ESERIES+5)
        right=eseries(other)
        if self.dims!=right.dims:
            raise TypeError("Incompatible operands for ESERIES addition")
        out=eseries()
        out.dims=self.dims
        out.shape=self.shape
        out.ampl=append(self.ampl,right.ampl)
        out.rates=append(self.rates,right.rates)
        return out
    def __radd__(self,other):#Addition with ESERIES on right (ex. 5+ESERIES)
        return self+other
    def __neg__(self):#define negation of ESERIES
        out=eseries()
        out.dims=self.dims
        out.shape=self.shape
        out.ampl=-self.ampl
        out.rates=self.rates
        return out 
    def __sub__(self,other):#Subtraction with ESERIES on left (ex. ESERIES-5)
        return self+(-other)
    def __rsub__(self,other):#Subtraction with ESERIES on right (ex. 5-ESERIES)
        return other+(-self)

    def __mul__(self,other):#Multiplication with ESERIES on left (ex. ESERIES*other)

        if isinstance(other,eseries):
            out=eseries()
            out.dims=self.dims
            out.shape=self.shape

            for i in range(len(self.rates)):
                for j in range(len(other.rates)):
                    out += eseries(self.ampl[i] * other.ampl[j], self.rates[i] + other.rates[j])

            return out
        else:
            out=eseries()
            out.dims=self.dims
            out.shape=self.shape
            out.ampl=self.ampl * other
            out.rates=self.rates
            return out

    def __rmul__(self,other): #Multiplication with ESERIES on right (ex. other*ESERIES)
        out=eseries()
        out.dims=self.dims
        out.shape=self.shape
        out.ampl=other * self.ampl
        out.rates=self.rates
        return out
    
    # 
    # todo:
    # select_ampl, select_rate: functions to select some terms given the ampl
    # or rate. This is done with {ampl} or (rate) in qotoolbox. we should use
    # functions with descriptive names for this.
    # 

    #   
    # evaluate the eseries for a list of times
    #
    def value(self, tlist):
        '''
        Evaluate an exponential series at the times listed in tlist. 
        '''

        #print "type tlist =", type(tlist)
        #print "len  ampl =", len(self.ampl)
        #print "type ampl =", type(self.ampl[0])

        if isinstance(tlist, float) or isinstance(tlist, int):
            tlist = [tlist]
    
        if isinstance(self.ampl[0], Qobj):
            # amplitude vector contains quantum objects
            val_list = []
        
            for j in range(len(tlist)):
                exp_factors = exp(array(self.rates) * tlist[j])
    
                val = 0
                for i in range(len(self.ampl)):
                    val += self.ampl[i] * exp_factors[i]
      
                val_list.append(val)
    
        else:
            # the amplitude vector contains c numbers
            val_list = zeros(size(tlist),dtype=complex)
    
            for j in range(len(tlist)):
                exp_factors = exp(array(self.rates) * tlist[j])
                val_list[j] = sum(dot(self.ampl, exp_factors))
    
    
        if len(tlist) == 1:
            return val_list[0]
        else:
            return val_list

    def spec(es, wlist):
        '''
        Evaluate the spectrum of an exponential series at frequencies in wlist. 
        '''
        val_list = zeros(size(wlist))

        for i in range(len(wlist)):
            val_list[i] = 2 * real( dot(es.ampl, 1./(1.0j*wlist[i] - es.rates)) )

        return val_list


    def tidy(self,*args):
        """ return a tidier version of self """

        #
        # combine duplicate entries (same rate)
        #
        rate_tol = 1e-10
        ampl_tol = 1e-10
 
        ampl_dict = {}
        unique_rates = {}
        ur_len = 0
    
        for r_idx in range(len(self.rates)):

            # look for a matching rate in the list of unique rates
            idx = -1
            for ur_key in unique_rates.keys():
                if abs(self.rates[r_idx] - unique_rates[ur_key]) < rate_tol:
                    idx = ur_key
                    break

            if idx == -1:
                # no matching rate, add it
                unique_rates[ur_len] = self.rates[r_idx]                
                ampl_dict[ur_len]    = [self.ampl[r_idx]]
                ur_len = len(unique_rates)
            else:
                # found matching rate, append amplitude to its list
                ampl_dict[idx].append(self.ampl[r_idx])

        # create new amplitude and rate list with only unique rates, and
        # nonzero amplitudes
        self.rates = array([])
        self.ampl  = array([])
        for ur_key in unique_rates.keys():
            total_ampl = sum(ampl_dict[ur_key])
            self.rates = append(self.rates, unique_rates[ur_key])
            self.ampl  = append(self.ampl, total_ampl)

        return self

#-------------------------------------------------------------------------------
#
# wrapper functions for accessing the class methods (for compatibility with
# quantum optics toolbox)
#
def esval(es, tlist):
    return es.value(tlist)

def esspec(es, wlist):
    return es.spec(wlist)

def estidy(es,*args):
    return es.tidy()


