from ..Qobj import *
from scipy import arange,prod,factorial,zeros,meshgrid

def qobjbasics():
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
    wignercat()












