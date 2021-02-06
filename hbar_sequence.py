#%%
from qutip.expect import expect
import numpy as np
from qutip.mesolve import mesolve
from qutip.operators import create
from qutip.qip.device.processor import Processor
from qutip.solver import Options
import time
from hbar_processor import HBAR_processor
from hbar_compiler import HBAR_Compiler
import hbar_fitting
from qutip import basis, fidelity
from qutip.qip.circuit import QubitCircuit
import matplotlib.pyplot as plt
from qutip.solver import Options
from qutip.operators import create, destroy, num, qeye
from tqdm import tqdm
#%%
#define system
qubit_dim=2
phonon_dim=2
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)

test_processor=HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26)
test_compiler = HBAR_Compiler(test_processor.num_qubits,\
     test_processor.params, test_processor.pulse_dict)

# %%
#phonon T1 measurement
def T1_measurement(t_list,processor,compiler):
    start_time = time.time()
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X", targets=0)
        circuit.add_gate('swap',targets=[0,1])
        circuit.add_gate('Wait',targets=0,arg_value=t)
        circuit.add_gate('swap',targets=[0,1])
        processor.load_circuit(circuit, compiler=compiler)
        option=Options()
        option.store_final_state=True
        result= test_processor.run_state(init_state = basis(dims, [0]+[0]*(phonon_num)),options=option)
        state=result.final_state
        qubit_measured_list.append(expect(num(processor.dims[0]),state.ptrace(0)))
    end_time = time.time()
    print('time used: ',end_time-start_time)
    plt.plot(t_list,qubit_measured_list)
    plt.show()
    return qubit_measured_list
#%% qubit_phonon oscillation
def phonon_rabi_measurement(t_list,processor,compiler):
    start_time = time.time()
    qubit_measured_list=[]
    for t in tqdm(t_list):
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X", targets=0)
        circuit.add_gate('Z_R',targets=[0,1],arg_value=t)
        processor.load_circuit(circuit, compiler=compiler)
        option=Options()
        option.store_final_state=True
        result= test_processor.run_state(init_state = basis(dims, [0]+[0]*(phonon_num)),options=option)
        state=result.final_state
        qubit_measured_list.append(expect(num(processor.dims[0]),state.ptrace(0)))
    end_time = time.time()
    print('time used: ',end_time-start_time)
    plt.plot(t_list,qubit_measured_list)
    plt.show()
    return qubit_measured_list

# %%
tl=np.linspace(0.1,10,100)
amp_list=phonon_rabi_measurement(tl,test_processor,test_compiler)
#%%
from importlib import reload  
reload(hbar_fitting)
t1_fitter=hbar_fitting.fitter(tl,np.array(amp_list))
t1_fitter.fit_phonon_rabi()
# %%

# %%
