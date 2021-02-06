
import numpy as np
from qutip.qip.compiler import Instruction
from qutip.qip.compiler import GateCompiler


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

def swap_phonon_compiler(gate, args):
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


def starkshift_compiler(gate, args):
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]
    g= parameters["g"]  # find the coupling strength for the target qubit
    gate_sigma =0.01
    amplitude =parameters['phonon_omega_z'][targets[1]-1]
    duration=gate.arg_value
    time_step=3e-3
    tlist = np.linspace(0, duration, int(duration/time_step))
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
    time_step=3e-3 #assume T1 at least 1us for each part of the system
    duration=gate.arg_value
    tlist=np.linspace(0,duration,int(duration/time_step))
    coeff=0*tlist
    pulse_info =[ ("Wait_T", coeff) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]

class HBAR_Compiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params, pulse_dict):
        super(HBAR_Compiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict)
        # pass our compiler function as a compiler for RX (rotation around X) gate.
        self.gate_compiler["X"] = gauss_rx_compiler
        self.gate_compiler["Wait"]=wait_complier
        self.gate_compiler["swap"]=swap_phonon_compiler
        self.gate_compiler['Z_R']=starkshift_compiler
        self.args.update({"params": params})