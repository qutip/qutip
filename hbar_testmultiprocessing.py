from importlib import reload
from qutip.metrics import process_fidelity  
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
import multiprocessing as mp
reload(hbar_fitting)
reload(hbar_processor)
reload(hbar_compiler)

qubit_dim=2
phonon_dim=10
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)

def run_circuit(circuit,processor,compiler,init_state=basis(dims, [0]+[0]*(phonon_num))):
    processor.load_circuit(circuit, compiler=compiler)
    option=Options()
    option.store_final_state=True
    option.store_states=False
    result=processor.run_state(init_state =init_state,options=option)
    state=result.final_state
    return state 

def T1_sequence(t):
    try:
        test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,rest_place=1)
        test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
            test_processor.params, test_processor.pulse_dict)
        circuit = QubitCircuit((phonon_num+1))
        circuit.add_gate("X_R", targets=0)
        circuit.add_gate('swap',targets=[0,1])
        circuit.add_gate('Wait',targets=0,arg_value=t)
        circuit.add_gate('swap',targets=[0,1])
        final_state=run_circuit(circuit,test_processor,test_compiler)
        return expect(num(test_processor.dims[0]),final_state.ptrace(0))
    except:
        return 'error happened'
if __name__ == '__main__':


    tl=np.linspace(0.1,10,1000)
    start=time.time()
    catch_result=[T1_sequence(t) for t in tl]
    end=time.time()
    print(end-start)
    nprocs = mp.cpu_count()
    print(f"Number of CPU cores: {nprocs}")

    start=time.time()
    pool = mp.Pool(processes=nprocs)
    catch_result=pool.map(T1_sequence,tl)

    pool.close()
    end=time.time() 
    print(end-start)
    print(catch_result)