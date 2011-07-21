from ..tensor import *
from ..Qobj import *
from ..states import *
from ..operators import *
from ..ptrace import *
from termpause import termpause

def tensorptrace():
    print '-'*80
    print 'Introduction to tensor products'
    print 'and partial traces in QuTiP.'
    print '-'*80
    termpause()
    print 'Tensor product of basis (Fock) states:'
    print '--------------------------------------'
    print 'tensor(basis(2,0), basis(2,0))'
    print tensor(basis(2,0), basis(2,0))
    
    print ''
    print 'tensor((basis(2,0)+basis(2,1)).unit(), (basis(2,0)+basis(2,1)).unit(), basis(2,0))'
    print tensor((basis(2,0)+basis(2,1)).unit(), (basis(2,0)+basis(2,1)).unit(), basis(2,0))
    
    print ''
    print 'Tensor product of operators:'
    print '----------------------------'
    print 'tensor(sigmax(), sigmax())'
    print tensor(sigmax(), sigmax())
    
    print ''
    print 'tensor(sigmaz(), qeye(2))'
    print tensor(sigmaz(), qeye(2))
    
    print ''
    print 'Composite Hamiltonians:'
    print '-----------------------'
    termpause()
    print 'Two coupled qubits:'
    print 'H = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz()) + 0.05 * tensor(sigmax(), sigmax())'
    print 'H'
    print tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz()) + 0.05 * tensor(sigmax(), sigmax())
    
    print ''
    print 'Thress coupled qubits:'
    print 'H = tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz()) + 0.5 * tensor(sigmax(), sigmax(), qeye(2)) +  0.25 * tensor(qeye(2), sigmax(), sigmax())'
    print 'H'
    print tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz()) + 0.5 * tensor(sigmax(), sigmax(), qeye(2)) +  0.25 * tensor(qeye(2), sigmax(), sigmax())
    
    print ''
    print 'Partial trace'
    print '-------------'
    termpause()
    
    print 'Single qubit state from composite two qubit state:'
    print 'psi = tensor(basis(2,0), basis(2,1))'
    print 'ptrace(psi, 0)'
    psi = tensor(basis(2,0), basis(2,1))
    print ptrace(psi, 0)
    
    print ''
    print 'ptrace(psi, 1)'
    print ptrace(psi, 1)
    
    print 'First qubit in superposition state:'
    print 'psi = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))'
    print 'psi'
    psi = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))
    print psi
    
    print ''
    print 'ptrace(psi, 0)'
    print ptrace(psi, 0)
    
    print ''
    print 'rho = tensor(ket2dm((basis(2,0)+basis(2,1)).unit()), fock_dm(2,0))'
    print 'rho'
    rho = tensor(ket2dm((basis(2,0)+basis(2,1)).unit()), fock_dm(2,0))
    print rho
    
    print ''
    print 'ptrace(rho, 0)'
    print ptrace(rho, 0)
    
    print ''
    print 'DEMO FINISHED...'
    termpause()
    
    
    








if __name__=="main()":
    tensorptrace()