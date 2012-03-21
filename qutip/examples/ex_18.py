#
# Entropy of binary system as probability
# of being in the excited state is varied.
#
from qutip.entropy import *
from qutip.states import *
from pylab import *

def run():
    a=linspace(0,1,20)
    out=zeros(len(a)) #preallocate output array
    for k in range(len(a)):
        # a*|0><0|
        x=a[k]*ket2dm(basis(2,0))
        # (1-a)*|1><1|
        y=(1-a[k])*ket2dm(basis(2,1))
        rho=x+y
        # Von-Neumann entropy (base 2) of rho
        out[k]=entropy_vn(rho,2)

    fig=figure()
    plot(a,out,lw=2)
    xlabel(r'Probability of being in excited state $(a)$')
    ylabel(r'Entropy')
    title("Entropy of $a|0\\rangle\langle0|+(1-a)|1\\rangle\langle1|$")
    show()

if __name__=='__main__':
    run()