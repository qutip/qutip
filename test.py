#########
#Monte-Carlo time evolution of an atom+cavity system.
#Adapted from a qotoolbox example by Sze M. Tan
#########
from qutip import *
from pylab import *
import time


N=5
a=coherent(N,.1+.1j)
print a


