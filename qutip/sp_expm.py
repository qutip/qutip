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
from scipy.linalg import norm,solve
import scipy.sparse as sp
from qobj import *

def sp_expm(qo):
    #############################
    def pade(m):
        n=shape(A)[0]
        c=padecoeff(m)
        if m!=13:
            apows= [[] for jj in xrange(int(ceil((m+1)/2)))]
            apows[0]=sp.eye(n,n).tocsr()
            apows[1]=A*A
            for jj in xrange(2,int(ceil((m+1)/2))):
                apows[jj]=apows[jj-1]*apows[1]
            U=sp.lil_matrix(zeros((n,n))).tocsr(); V=sp.lil_matrix(zeros((n,n))).tocsr()
            for jj in xrange(m,0,-2):
                U=U+c[jj]*apows[jj/2]
            U=A*U
            for jj in xrange(m-1,-1,-2):
                V=V+c[jj]*apows[(jj+1)/2]
            F=solve((-U+V).todense(),(U+V).todense())
            return sp.lil_matrix(F).tocsr()
        elif m==13:
            A2=A*A
            A4=A2*A2
            A6=A2*A4
            U = A*(A6*(c[13]*A6+c[11]*A4+c[9]*A2)+c[7]*A6+c[5]*A4+c[3]*A2+c[1]*sp.eye(n,n).tocsr())
            V = A6*(c[12]*A6 + c[10]*A4 + c[8]*A2)+ c[6]*A6 + c[4]*A4 + c[2]*A2 + c[0]*sp.eye(n,n).tocsr()
            F=solve((-U+V).todense(),(U+V).todense()) 
            return sp.lil_matrix(F).tocsr()
    #################################
    A=qo.data #extract Qobj data (sparse matrix)
    m_vals=array([3,5,7,9,13])
    theta=array([0.01495585217958292,0.2539398330063230,0.9504178996162932,2.097847961257068,5.371920351148152],dtype=float)
    normA=norm(A.todense(),1)
    if normA<=theta[-1]:
        for ii in xrange(len(m_vals)):
            if normA<=theta[ii]:
                F=pade(m_vals[ii])
                break
    else:
        t,s=frexp(normA/theta[-1])
        s=s-(t==0.5)
        A=A/2.0**s
        F=pade(m_vals[-1])
        for i in xrange(s):
            F=F*F
    return qobj(F,dims=qo.dims,shape=qo.shape)
        

def padecoeff(m):
    if m==3:
        return array([120, 60, 12, 1])
    elif m==5:
        return array([30240, 15120, 3360, 420, 30, 1])
    elif m==7:
        return array([17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1])
    elif m==9:
        return array([17643225600, 8821612800, 2075673600, 302702400, 30270240,2162160, 110880, 3960, 90, 1])
    elif m==13:
        return array([64764752532480000, 32382376266240000, 7771770303897600,1187353796428800, 129060195264000, 10559470521600,670442572800, 33522128640, 1323241920,40840800, 960960, 16380, 182, 1])







