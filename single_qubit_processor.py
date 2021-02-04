#%%
from qutip.qobj import dims
from qutip.operators import create, destroy, num
import numpy as np
import qutip
from qutip import sigmax, sigmay, sigmaz, tensor, fidelity
from qutip.qip.pulse import Pulse
from qutip.qip.device.processor import Processor
import matplotlib.pyplot as plt


#%%
class single_qubit_processor(Processor):
    def __init__(self,N,t1,t2,dims,h_x,h_z,alpha):
        super(single_qubit_processor,self).__init__(N,t1,t2,dims)
        self.set_up_params(N,h_x,h_z,alpha)
        self.set_up_ops()
        self.set_up_drift() 

    def set_up_params(self,N, h_x, h_z, alpha):
        self.params = {}
        self.params["sz"] = [2 * np.pi * h_z] * N  # for each qubtis, they can also be different
        self.params["sx"] = [2 * np.pi * h_x] * N
        self.params["alpha"] = [2 * np.pi * alpha] * N
        # Here goes all computation of hardware parameters. They all need to be saved in self.params for later use.
        # The computed parameters can be used e.g. in setting up the Hamiltonians or the compiler to compute the pulse coefficients.

    def set_up_ops(self):
        self.pulse_dict = {}  # A dictionary that maps the name of a pulse to its index in the pulse list.
        index = 0
        # sx_ops
        for m in range(self.N):
            self.add_control(create(self.dims[m])+destroy(self.dims[m]), m, label="sx" + str(m)) # sigmax pulse on m-th qubit with the corresponding pulse 
            self.pulse_dict["sx" + str(m)] = index
            index += 1
        # sz_ops
        for m in range(self.N):
            self.add_control(num(self.dims[m]), m, label="sz" + str(m))
            self.pulse_dict["sz" + str(m)] = index
            index += 1
    def set_up_drift(self):
        for i in range(self.N):
            self.add_drift(create(self.dims[i])**2*destroy(self.dims[i])**2,i)

    def load_circuit(self, circuit, schedule_mode=False, compiler=None):
        # resolve the circuit to a certain basis
        resolved_circuit = circuit.resolve_gates(basis=["ISWAP", "RX", "RZ"])  # other supported gates includes RY, Cnum_qubitsOT, SQRTISWAP
        # compile the circuit to control pulses
        
        tlist, coeffs = compiler.compile(resolved_circuit, schedule_mode=schedule_mode)
        # save the time sequence and amplitude for all pulses
        self.set_all_tlist(tlist)
        self.coeffs = coeffs
        return tlist, self.coeffs

# %%
qubits_num=2
test_processor=single_qubit_processor(qubits_num,[20,20],[39,39],[3,3],0.03,0.03,150)
from qutip import basis
from qutip.qip.circuit import QubitCircuit
circuit = QubitCircuit(qubits_num)
circuit.add_gate("X", targets=1)
circuit.add_gate("X", targets=0)
circuit.png

# %%
from qutip.qip.compiler import Instruction
def gauss_dist(t, sigma, amplitude, duration):
    return amplitude/np.sqrt(2*np.pi) /sigma*np.exp(-0.5*((t-duration/2)/sigma)**2)

def gauss_rx_compiler(gate, args):
    """
    Compiler for the RX gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    h_x2pi = parameters["sx"][targets[0]]  # find the coupling strength for the target qubit
    amplitude = gate.arg_value / 2. / 0.9973 #  0.9973 is just used to compensate the finite pulse duration so that the total area is fixed
    gate_sigma = h_x2pi / np.sqrt(2*np.pi)
    duration = 6 * gate_sigma
    tlist = np.linspace(0, duration, 100)
    coeff = gauss_dist(tlist, gate_sigma, amplitude, duration)
    pulse_info = [("sx" + str(targets[0]), coeff)]  #  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

from qutip.qip.compiler import GateCompiler
class MyCompiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params, pulse_dict):
        super(MyCompiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["RX"] = gauss_rx_compiler
        self.args.update({"params": params})
# %%
gauss_compiler = MyCompiler(test_processor.num_qubits,\
     test_processor.params, test_processor.pulse_dict)
tlist, coeff = test_processor.load_circuit(circuit, compiler=gauss_compiler)
result = test_processor.run_state(init_state = basis([2,2], [0,0]))
print("fidelity without scheduling:", fidelity(result.states[-1], basis([2,2],[1,1])))
# %%
fidelity_list=[]
for state in result.states:
    fidelity_list.append( fidelity(state, basis([2,2],[1,1])) )
plt.plot(result.times,fidelity_list)
# %%
test_processor.plot_pulses()
# %%
