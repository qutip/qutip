from qutip.expect import expect
import numpy as np

from qutip.solver import Options
from qutip import basis
from qutip.qip.circuit import QubitCircuit
from qutip.solver import Options
from qutip.operators import num
from tqdm import tqdm


def run_circuit(circuit,processor,compiler,init_state=None):
    processor.load_circuit(circuit, compiler=compiler)
    option=Options()
    option.store_final_state=True
    option.store_states=False
    if not init_state:
        init_state=basis(processor.dims, [0]+[0]*(processor.N-1))
    result=processor.run_state(init_state =init_state,options=option)
    state=result.final_state
    return state 

#phonon T1 measurement
def phonon_T1_measurement(t_list,processor,compiler):
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((processor.N-1+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('swap',targets=[0,1])
        circuit.add_gate('Wait',targets=0,arg_value=t)
        circuit.add_gate('swap',targets=[0,1])
        final_state=run_circuit(circuit,processor,compiler)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list


#qubit_phonon oscillation measurement
def phonon_rabi_measurement(t_list,processor,compiler,swap_time_list=[]):
    prepared_state=basis( processor.dims, [0]+[0]*(processor.N-1))
    for swap_t in swap_time_list:
        circuit = QubitCircuit((processor.N))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
        prepared_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((processor.N-1+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=t)
        final_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list

# qubit spectrum measurement
def qubit_pec_measurement(detuning_list,processor,compiler,params):
    qubit_measured_list=[]
    for detuning in tqdm(detuning_list):
        params['detuning']=detuning 
        circuit = QubitCircuit((processor.N-1+1))   
        circuit.add_gate("XY_R_GB", targets=0,arg_value=params)
        final_state=run_circuit(circuit,processor,compiler)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list

# fock state phonon number splitting
def num_split_fock_measurement(detuning_list,processor,compiler,params,swap_time_list=[]):
    #prepare state
    prepared_state=basis( processor.dims, [0]+[0]*(processor.N-1))
    for swap_t in swap_time_list:
        circuit = QubitCircuit((processor.N))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
        prepared_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
    #probe qubit 
    qubit_measured_list=[]
    for detuning in tqdm(detuning_list):
        params['detuning']=detuning 
        circuit = QubitCircuit((processor.N-1+1))   
        circuit.add_gate("XY_R_GB", targets=0,arg_value=params)
        final_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list

#coherent state number splitting
def num_split_coh_measurement(detuning_list,processor,compiler,param_phonon_drive,param_probe):
    #prepare state
    prepared_state=basis( processor.dims, [0]+[0]*(processor.N-1))
    circuit = QubitCircuit((processor.N-1+1))    
    circuit.add_gate("XY_R_GB", targets=0,arg_value=param_phonon_drive)
    prepared_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
    #probe qubit 
    qubit_measured_list=[]
    for detuning in tqdm(detuning_list):
        param_probe['detuning']=detuning 
        circuit = QubitCircuit((processor.N-1+1))   
        circuit.add_gate("XY_R_GB", targets=0,arg_value=param_probe)
        final_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list

