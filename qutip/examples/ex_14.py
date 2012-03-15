#
# Prove that 1-F**2 <= T for pure state density matricies
# where F and T are the fidelity and trace distance metrics,
# respectively using randomly generated ket vectors.
#
from ..metrics import *
from ..rand import *
from.. states import *
from pylab import *

def run():
    N=21#number of kets to generate
    
    #create arrays of pure density matrices from random kets using ket2dm
    x=array([ket2dm(rand_ket(10)) for k in range(N)])
    y=array([ket2dm(rand_ket(10)) for k in range(N)])
    
    #calculate trace distance and fidelity between states in x & y
    T=array([tracedist(x[k],y[k]) for k in range(N)])
    F=array([fidelity(x[k],y[k]) for k in range(N)])

    #plot T and 1-F**2 where x=range(N)
    plot(range(N),T,'b',range(N),1-F**2,'r',lw=2)
    title("Verification of 1-F**2<=T for random pure states.")
    legend(("trace distance","1-fidelity**2"),loc=0)
    show()

