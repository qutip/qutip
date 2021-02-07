from qutip.operators import create, destroy, num, qeye
import numpy as np
from qutip import sigmax, sigmay, sigmaz, tensor, fidelity
from qutip.qip.device.processor import Processor



class HBAR_processor(Processor):
    def __init__(self,N,t1,t2,dims,Omega=20,alpha=200,FSR=13,g=0.266,rest_place=6.5):
        super(HBAR_processor,self).__init__(N,t1,t2,dims)
        self.set_up_params(Omega,alpha,FSR,g)
        self.set_up_ops()
        self.set_up_drift() 

    def set_up_params(self, Omega,alpha,FSR,g,rest_place):
        self.params = {}
        self.params["Omega"] = 2*np.pi*Omega  # default rabi frequency
        self.params["alpha"] = 2*np.pi*alpha  # enharmonic term
        self.params['phonon_omega_z']=(np.array(range(self.N-1))*FSR-rest_place)*2*np.pi
        self.params['g']=g*2*np.pi
        # Here goes all computation of hardware parameters. They all need to be saved in self.params for later use.
        # The computed parameters can be used e.g. in setting up the Hamiltonians or the compiler to compute the pulse coefficients.

    def set_up_ops(self):
        self.pulse_dict = {}  # A dictionary that maps the name of a pulse to its index in the pulse list.
        index = 0
        
        # X-axis rotate
        self.add_control(create(self.dims[0])+destroy(self.dims[0]), 0, label="X-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["X-axis_R"] = index
        index += 1
        
        # Y-axis rotate
        self.add_control(1j*create(self.dims[0])-1j*destroy(self.dims[0]), 0, label="Y-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Y-axis_R"] = index
        index += 1
        
        # Z-axis rotate
        self.add_control(num(self.dims[0]), 0, label="Z-axis_R") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Z-axis_R"] = index
        index += 1

        #wait
        self.add_control(0*qeye(self.dims[0]), 0, label="Wait_T") # sigmax pulse on m-th qubit with the corresponding pulse 
        self.pulse_dict["Wait_T"] = index
        index +=1

    def set_up_drift(self):
        #qubit enharmonic term
        self.add_drift(self.params['alpha']*create(self.dims[0])**2*destroy(self.dims[0])**2,0)
        #qubit phonon coupling
        for i in range(self.N-1):
            self.add_drift(
                self.params['g']*(
                    tensor(create(self.dims[0]),destroy(self.dims[i+1]))\
                        +tensor(destroy(self.dims[0]),create(self.dims[i+1]))
                        ),[0,i+1]
                            )

        #add phonon frequency
        for i in range(self.N-1):
            self.add_drift(
                self.params['phonon_omega_z'][i]*(num(self.dims[i+1])
                        ),i+1
                            )
    def load_circuit(self, circuit, schedule_mode=False, compiler=None):
        # resolve the circuit to a certain basis
        # circuit = circuit.resolve_gates(basis=["ISWAP", "RX", "RZ"])  # other supported gates includes RY, Cnum_qubitsOT, SQRTISWAP
        # compile the circuit to control pulses
        
        tlist, coeffs = compiler.compile(circuit, schedule_mode=schedule_mode)
        # save the time sequence and amplitude for all pulses
        self.set_all_tlist(tlist)
        self.coeffs = coeffs
        return tlist, self.coeffs



