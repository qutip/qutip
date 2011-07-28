from ..states import *
from ..Qobj import *
from ..tensor import *
from ..ptrace import *
from ..operators import *
from ..expect import *
from ..correlation import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from termpause import termpause


#
# run the example
#
def esdemo():

    print "== Demonstration of exponential series (time-dependent quantum objects) === "
   
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Example eseries object: sigmax() * exp(i * omega * t)
    ---
    omega = 1.0
    print eseries(sigmax(), 1j * omega)
    """
    omega = 1.0
    print eseries(sigmax(), 1j * omega)
    
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Example eseries object: sigmax() * cos(omega * t)
    ---
    omega = 1.0
    print eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    """
    omega = 1.0
    print eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Evaluate eseries object at time t = 0.0
    ---
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    print esval(es, 0.0)
    """
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    print esval(es, 0.0)
    
    
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Evaluate eseries object at array of times t = [0.0, 1.0 * pi, 2.0 * pi]
    ---
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    tlist = [0.0, 1.0 * pi, 2.0 * pi]
    print esval(es2, tlist)
    """
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    tlist = [0.0, 1.0 * pi, 2.0 * pi]
    print esval(es, tlist)
    
    
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Expectation values of eseries
    ---
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    tlist = [0.0, 1.0 * pi, 2.0 * pi]

    print expect(sigmax(), es)
    """
    omega = 1.0
    es = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
    tlist = [0.0, 1.0 * pi, 2.0 * pi]

    print expect(sigmax(), es)
    
    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Arithmetics with eseries
    ---
    es1 = eseries(sigmax(), 1j * omega)
    print "es1 =", es1    
    es2 = eseries(sigmax(), -1j * omega)
    print "es2 =", es2
    
    print "===> es1 + es2 ="
    print es1 + es2
    print "===> es1 - es2 ="
    print es1 - es2
    print "===> es1 * es2 ="
    print es1 * es2
    print "===> (es1 + es2) * (es1 - es2) ="
    print (es1 + es2) * (es1 - es2)
    """
    es1 = eseries(sigmax(), 1j * omega)
    print "es1 =", es1    
    es2 = eseries(sigmax(), -1j * omega)
    print "es2 =", es2
    
    print "===> es1 + es2 ="
    print es1 + es2
    print "===> es1 - es2 ="
    print es1 - es2
    print "===> es1 * es2 ="
    print es1 * es2
    print "===> (es1 + es2) * (es1 - es2) ="
    print (es1 + es2) * (es1 - es2)
    

    # --------------------------------------------------------------------------
    termpause()
    print """
    --------------------------------------------------------------------------------
    ---
    --- Expectation values of eseries: sigmaz() * cos(omega * t) + sigmax() * sin(omega * t)
    ---
    """
    es = eseries([0.5*sigmaz(), 0.5*sigmaz()], [1j, -1j]) + eseries([-0.5j*sigmax(), 0.5j*sigmax()], [1j, -1j])
    print "es =\n", es

    print "es at t=0.0  =\n", es.value(0.0)
    print "es at t=pi/2 =\n", es.value(pi/2)

    rho = fock_dm(2, 1)
    es3_expect = expect(rho, es)
    print "Expectation value of es for excited spin state ="
    print es3_expect

    print "Expectation value at t = 0 and t = pi/2"
    print es3_expect.value([0.0, pi/2])


if __name__=='main()':
    esdemo()

