#%%
from importlib import reload  
from qutip.expect import expect
import numpy as np
from qutip.mesolve import mesolve
from qutip.operators import create
from qutip.qip import circuit
from qutip.qip.device.processor import Processor
from qutip.solver import Options
import time
import hbar_processor 
import hbar_compiler 
import hbar_fitting
from qutip import basis, fidelity
from qutip.qip.circuit import QubitCircuit
import matplotlib.pyplot as plt
from qutip.solver import Options
from qutip.operators import create, destroy, num, qeye
from tqdm import tqdm
#%%
reload(hbar_fitting)
reload(hbar_processor)
reload(hbar_compiler)
#%%
#define system
qubit_dim=2
phonon_dim=10
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)

test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,rest_place=1)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
     test_processor.params, test_processor.pulse_dict)

def run_circuit(circuit,processor,compiler,init_state=basis(dims, [0]+[0]*(phonon_num))):
    processor.load_circuit(circuit, compiler=compiler)
    option=Options()
    option.store_final_state=True
    result= test_processor.run_state(init_state =init_state,options=option)
    state=result.final_state
    return state 
    

# %%
#phonon T1 measurement
def phonon_T1_measurement(t_list,processor,compiler):
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('swap',targets=[0,1])
        circuit.add_gate('Wait',targets=0,arg_value=t)
        circuit.add_gate('swap',targets=[0,1])
        final_state=run_circuit(circuit,processor,compiler)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list

#%% qubit_phonon oscillation
def phonon_rabi_measurement(t_list,processor,compiler,swap_time_list=[]):
    prepared_state=basis(dims, [0]+[0]*(phonon_num))
    for swap_t in swap_time_list:
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
        prepared_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=t)
        final_state=run_circuit(circuit,processor,compiler,init_state=prepared_state)
        qubit_measured_list.append(expect(num(processor.dims[0]),final_state.ptrace(0)))
    return qubit_measured_list
swap_t_list_simulated=[0.9615710875211836, 0.6793394959515644, 0.5549226661382177, 0.4804636060930446,\
     0.4294620578370378, 0.3923078531720593, 0.3639007000694595, 0.34220291598663793]
# %%
# higher order qubit_phonon oscillation
if 0:
    tl=np.linspace(0.1,10,100)
    swap_t_list=[]
    for i in range(8):
        amp_list=phonon_rabi_measurement(tl,test_processor,test_compiler,swap_time_list=swap_t_list)
        plt.plot(tl,amp_list)
        t1_fitter=hbar_fitting.fitter(tl,np.array(amp_list))
        swap_t=t1_fitter.fit_phonon_rabi()
        swap_t_list.append(swap_t)
# %%
#qubit spec
detuning_list=np.linspace(-0.5,1,151)
# detuning_list=[0]
fidelity_list=[]
for detuning in tqdm(detuning_list):
    params={'Omega':0.0125,
    'phase':0,
    'sigma':0.02,
    'duration':20,
    'detuning':detuning
    }
    circuit = QubitCircuit((phonon_num+1))    
    for swap_t in swap_t_list_simulated:
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
    circuit.add_gate("XY_R_GB", targets=0,arg_value=params)
    test_processor.load_circuit(circuit, compiler=test_compiler)
    result= test_processor.run_state(init_state =basis(dims, [0]+[0]*(phonon_num)))
    fidelity_list.append(expect(num(test_processor.dims[0]),result.states[-1].ptrace(0)))

plt.plot(detuning_list,fidelity_list)
plt.show()
# %%
fidelity_list=[]
for state in result.states:
    fidelity_list.append(expect(num(test_processor.dims[0]),state.ptrace(0)))
plt.plot(result.times,fidelity_list)

# %%
