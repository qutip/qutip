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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from ..Qobj import *
from ..istests import *
from ..states import *
from scipy import arange,prod,factorial,zeros,meshgrid
from termpause import termpause

def qobjbasics():
    print 'Basic Qobj usage and examples.'
    termpause()
    print "======================================"
    A = Qobj([0.8, 0.1, 0.1, 0.2])
    print "A = \n", A
    print "A isbra  = ", isbra(A)
    print "A isket  = ", isket(A)
    print "A isoper = ", isoper(A)
    print "A isherm = ", A.isherm
    print "A len    = ", prod(A.shape)
    print "iter     = ", arange(0, prod(A.shape))

    print "======================================"
    A = Qobj([[0.8, 0.1], [0.1, 0.2]])
    print "A = \n", A
    print "A isket  = ", isket(A)
    print "A isoper = ", isoper(A)
    print "A isherm = ", A.isherm
    print "A len    = ", prod(A.shape)
    print "iter     = ", arange(0, prod(A.shape))


    print ""

    print "======================================"
    A = Qobj([0])
    print "A = \n", A
    print "A isket  = ", isket(A)
    print "A isoper = ", isoper(A)
    print "A isherm = ", A.isherm
    print "A len    = ", prod(A.shape)
    print "iter     = ", arange(0, prod(A.shape))
    print 'type     = ', A.type
    print "======================================"
    X,Y = meshgrid(array([0,1,2]), array([0,1,2]))
    Z   = zeros(size(X))
    print "X = \n", X
    print "X size = \n", size(X)
    print "Z = \n", Z
    print "Z size = \n", size(Z)
    print

    print "======================================"
    psi = basis(4, 2)
    print "basis(4,2) = \n", psi
    print " isket     =  ",isket(psi)
    print "psi isherm = ", psi.isherm

    print "======================================"
    psi = arange(0, 30)
    print "psi = \n", factorial(psi)



if __name__=='main()':
    qobjbasics()












