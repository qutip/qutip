#%%
from operator import le
from qutip.mesolve import mesolve
from qutip.operators import create, destroy, num, qeye
import numpy as np
import qutip as qt
from qutip import sigmax, sigmay, sigmaz, tensor, fidelity
from qutip.qip.pulse import Pulse
from qutip.qip.device.processor import Processor
import matplotlib.pyplot as plt
from qutip.qip.compiler import Instruction
from qutip.qip.compiler import GateCompiler
from qutip import basis
from qutip.qip.circuit import QubitCircuit
import time

#%%


class single_qubit_processor(Processor):
    def __init__(self,N,t1,t2,dims,Omega=20,alpha=200,FSR=13,g=0.3):
        super(single_qubit_processor,self).__init__(N,t1,t2,dims)
        self.set_up_params(Omega,alpha,FSR,g)
        self.set_up_ops()
        self.set_up_drift() 

    def set_up_params(self, Omega,alpha,FSR,g):
        self.params = {}
        self.params["Omega"] = 2*np.pi*Omega  # for each qubtis, they can also be different
        self.params["alpha"] = 2*np.pi*alpha
        self.params['phonon_omega_z']=(np.array(range(self.N-1))*FSR-FSR/2)*2*np.pi
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
                self.params['phonon_omega_z'][i]*(num(dims[i+1])
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

#define gates

#define gauss block shape
def gauss_block(t,sigma,amplitude,duration):
    if t<2*sigma:
        return amplitude*np.exp(-0.5*((t-2*sigma)/sigma)**2)
    elif t>duration-2*sigma:
        return amplitude*np.exp(-0.5*((t+2*sigma-duration)/sigma)**2)
    else:
        return amplitude
gauss_block=np.vectorize(gauss_block)
#define sigma_z gaussian block shape pulse
#define gaussian shape x rotation, normally for quick pulse
def swap_phonon(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    g= parameters["g"]  # find the coupling strength for the target qubit
    gate_sigma =0.01
    amplitude =parameters['phonon_omega_z'][targets[1]-1]
    duration = 1/g/4*2*np.pi
    tlist = np.linspace(0, duration, 1000)
    coeff = gauss_block(tlist,gate_sigma, amplitude, duration)
    pulse_info =[ ("Z-axis_R", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]


#define gauss shape 
def gauss_dist(t, sigma, amplitude, duration):
    return amplitude*np.exp(-0.5*((t-duration/2)/sigma)**2)

#define gaussian shape x rotation, normally for quick pulse
def gauss_rx_compiler(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    Omega= parameters["Omega"]  # find the coupling strength for the target qubit
    gate_sigma = 1/Omega
    amplitude = Omega/2.49986*np.pi/2 #  0.9973 is just used to compensate the finite pulse duration so that the total area is fixed
    duration = 6 * gate_sigma
    tlist = np.linspace(0, duration, 300)
    coeff = gauss_dist(tlist, gate_sigma, amplitude, duration)
    pulse_info =[ ("X-axis_R", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

#define waiting pulse
def wait_complier(gate,args):
    targets = gate.targets  # target qubit
    parameters = args["params"]
    time_step=1e-3 #assume T1 at least 1us for each part of the system
    duration=gate.arg_value
    tlist=np.linspace(0,duration,int(duration/time_step))
    coeff=0*tlist
    pulse_info =[ ("Wait_T", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

class MyCompiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params, pulse_dict):
        super(MyCompiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["X"] = gauss_rx_compiler
        self.gate_compiler["Wait"]=wait_complier
        self.gate_compiler["swap"]=swap_phonon
        self.args.update({"params": params})
# %%
qubit_dim=2
phonon_dim=2
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[20]+[60]*(phonon_num)
t2=[30]+[100]*(phonon_num)

test_processor=single_qubit_processor((phonon_num+1),t1,t2,dims)
start_time = time.time()

circuit = QubitCircuit((phonon_num+1))
circuit.add_gate("X", targets=0)
circuit.add_gate('swap',targets=[0,1])
circuit.add_gate('Wait',targets=0,arg_value=1)
circuit.add_gate('swap',targets=[0,1])

gauss_compiler = MyCompiler(test_processor.num_qubits,\
     test_processor.params, test_processor.pulse_dict)
tlist, coeff = test_processor.load_circuit(circuit, compiler=gauss_compiler)
result = test_processor.run_state(init_state = basis(dims, [0]+[0]*(phonon_num)))

end_time = time.time()
print(end_time - start_time)
# %%
if 1:
    fidelity_list=[]
    for state in result.states:
        fidelity_list.append( fidelity( state.ptrace(0), basis([dims[0]],[1])))
        # fidelity_list.append( fidelity( state.ptrace(0),create(3)+destroy(3) ))

    plt.plot(result.times,fidelity_list)
    plt.show()

