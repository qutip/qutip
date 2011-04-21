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
from scipy import *
from qobj import *

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
                #sh=array([qobj(x).shape for x in range(0,num)])
                sh=array([qobj(x).shape for x in q])
                if any(sh!=sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                #self.ampl=array([qobj(x) for x in q])
                self.ampl=array([x for x in q])
                self.rates=zeros(ind)
                self.dims=self.ampl[0].dims
                self.shape=self.ampl[0].shape
            elif isinstance(q,qobj):
                qo=qobj(q)
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
                sh=array([qobj(q[x]).shape for x in range(0,num)])
                if any(sh!=sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl=array([qobj(q[x]) for x in range(0,num)])
                self.dims=self.ampl[0].dims
                self.shape=self.ampl[0].shape
            else:
                num=1
                self.ampl=array([qobj(q)])
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


def esval(es, tlist):
    '''
    Evaluate an exponential series at the times listed in tlist. 
    '''
    #val_list = [] #zeros(size(tlist))
    val_list = zeros(size(tlist))

    for j in range(len(tlist)):
        exp_factors = exp(array(es.rates) * tlist[j])

        #val = 0
        #for i in range(len(es.ampl)):
        #    val += es.ampl[i] * exp_factors[i]
        val_list[j] = sum(dot(es.ampl, exp_factors))
  
        #val_list[j] = val
        #val_list.append(val)

    return val_list


def esspec(es, wlist):
    '''
    Evaluate the spectrum of an exponential series at frequencies in wlist. 
    '''

    val_list = zeros(size(wlist))

    for i in range(len(wlist)):
        
        #print "data =", es.ampl
        #print "demon =", 1/(1j*wlist[i] - es.rates)

        val_list[i] = 2 * real( dot(es.ampl, 1/(1j*wlist[i] - es.rates)) )

    return val_list


##########---ESERIES TIDY---#############################
def estidy(es,*args):
    out=eseries()
    #zipped=zip(es.rates,es.ampl)#combine arrays so that they can be sorted together
    #zipped.sort() #sort rates from lowest to highest
    out.rates = [] 
    out.ampl  = []
    out.dims  = es.ampl[0].dims
    out.shape = es.ampl[0].shape

    #
    # determine the tolerance
    # 
    if not any(args):
        tol1=array([1e-6,1e-6])
        tol2=array([1e-6,1e-6])
    elif len(args)==1:
        if len(args[0])==1:
            tol1[1]=0
        tol2=array([1e-6,1e-6])
    elif len(args)==2:
        if len(args[1])==1:
            tol2[1]=0
    rates=es.rates
    rmax=max(abs(array(rates)))
    rlen=len(es.rates)
    data=es.ampl
    tol=max(tol1[0]*rmax,tol1[1])

    #
    # find unique rates (todo: allow deviations within tolerance)
    #
    rates_unique = sort(list(set(rates)))

    #
    # collect terms that have the same rates (within the tolerance)
    #
    for r in rates_unique:
    
        terms = qobj() 

        for idx,rate in enumerate(rates):
            if abs(rate - r) < tol:
                terms += es.ampl[idx]

    	if terms.norm() > tol:
            out.rates.append(r)
    	    out.ampl.append(terms)
 
    return out


###########---Find Groups---####################
def findgroups(values,index,tol):
    zipped=zip(values,index)#combine arrays so that they can be sorted together
    zipped.sort() #sort rates from lowest to highest
    vs,vperm=zip(*zipped) 
    big=where(diff(vs)>tol,1,0)
    sgroup=append(array([1]),big)
    sindex=array(vperm)
    return sindex,sgroup



