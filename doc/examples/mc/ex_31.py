#
# Occupation number of two coupled osciilators with
# oscillator A driven by an external classical drive.
# Both oscillators are assumed to start in the ground
# state.
#
from qutip import *
from pylab import *

def run():
    wa  = 1.0 * 2 * pi   # frequency of system a
    wb  = 1.0 * 2 * pi   # frequency of system a
    wab = 0.2 * 2 * pi   # coupling frequency
    ga = 0.2 * 2 * pi    # dissipation rate of system a
    gb = 0.1 * 2 * pi    # dissipation rate of system b
    Na = 10              # number of states in system a
    Nb = 10              # number of states in system b
    E = 1.0 * 2 * pi     # Oscillator A driving strength 

    a = tensor(destroy(Na), qeye(Nb))
    b = tensor(qeye(Na), destroy(Nb))
    na = a.dag() * a
    nb = b.dag() * b
    H = wa*na + wb*nb + wab*(a.dag()*b+a*b.dag()) + E*(a.dag()+a)

    # start with both oscillators in ground state
    psi0 = tensor(basis(Na), basis(Nb))

    c_op_list = []
    c_op_list.append(sqrt(ga) * a)
    c_op_list.append(sqrt(gb) * b)

    tlist = linspace(0, 5, 101)

    #run simulation
    data = mcsolve(H,psi0,tlist,c_op_list,[na,nb])

    #plot results
    plot(tlist,data.expect[0],'b',tlist,data.expect[1],'r',lw=2)
    xlabel('Time',fontsize=14)
    ylabel('Excitations',fontsize=14)
    legend(('Oscillator A', 'Oscillator B'))
    show()
    
if __name__=='__main__':
    run()