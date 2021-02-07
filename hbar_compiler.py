
import numpy as np
from qutip.qip.compiler import Instruction
from qutip.qip.compiler import GateCompiler
import matplotlib.pyplot as plt

#define gauss block shape
def gauss_block(t,sigma,amplitude,duration):
    if t<2*sigma:
        return amplitude*np.exp(-0.5*((t-2*sigma)/sigma)**2)
    elif t>duration-2*sigma:
        return amplitude*np.exp(-0.5*((t+2*sigma-duration)/sigma)**2)
    else:
        return amplitude
gauss_block=np.vectorize(gauss_block)

# define modulated gauss block shape
def gauss_block_modulated(t,sigma,amplitude,duration,omega,phase):
    return gauss_block(t,sigma,amplitude,duration)*np.cos(omega*t+phase)

#define gauss_block shape modulated which used for qubit XY plane rotation, mostly used 
def gauss_block_modulated_XY_rotate_compiler(gate,args):   
    """
    Compiler for the X-axis_Rotate gate
    """
    targets = gate.targets  # target qubit
    parameters = args["params"]

    #set pulse strength (rabi frequency)
    if gate.arg_value['Omega']:
        Omega=gate.arg_value['Omega']*np.pi*2
    else:
        Omega= parameters["Omega"]
    #set pulse raising time (sigma)
    if gate.arg_value['sigma']:
        gate_sigma=gate.arg_value['sigma']
    else:
        gate_sigma = 0.01
    #set drive detuning frequency
    if gate.arg_value['detuning']:
        omega=gate.arg_value['detuning']*np.pi*2
    else:
        omega=0
    #set drive direction
    if gate.arg_value['phase']:
        phase=gate.arg_value['phase']
    else:
        phase=0
    #set pulse duration
    duration=gate.arg_value['duration']
    
    time_step=3e-3
    tlist = np.linspace(0, duration, int(duration/time_step))
    coeff1 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,omega, phase)
    coeff2 = gauss_block_modulated(tlist, gate_sigma, Omega, duration,omega, phase+np.pi/2)

    pulse_info =[ ("X-axis_R", coeff1),("Y-axis_R",coeff2) ]#  save the information in a tuple (pulse_name, coeff)
    return [Instruction(gate, tlist, pulse_info)]


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

#define sigma_z gaussian block shape pulse
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
    if gate.arg_value:
        Omega=gate.arg_value['Omega']
    else:
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
        self.gate_compiler["X_R"] = gauss_rx_compiler
        self.gate_compiler["Wait"]=wait_complier
        self.gate_compiler["swap"]=swap_phonon_compiler
        self.gate_compiler['Z_R_GB']=starkshift_compiler
        self.gate_compiler['XY_R_GB']= gauss_block_modulated_XY_rotate_compiler
        self.args.update({"params": params})