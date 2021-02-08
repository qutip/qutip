import hbar_compiler
import hbar_processor
import hbar_sequence
import hbar_fitting
import numpy as np
import matplotlib.pyplot as plt

qubit_dim=2
phonon_dim=10
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)
test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,rest_place=1)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)

#fitted simulation result of swap gate time for each fock state.
swap_t_list_simulated=np.array([0.9615710875211836, 0.6793394959515644, 0.5549226661382177, 0.4804636060930446,\
     0.4294620578370378, 0.3923078531720593, 0.3639007000694595, 0.34220291598663793])

t_L=np.linspace(0.1,10,100)
detuning_L=np.linspace(-0.2,0.5,71)
param1={'Omega':0.025,
    'phase':0,
    'sigma':0.02,
    'duration':10,
    }
catch_result=hbar_sequence.num_split_fock_measurement(detuning_L,test_processor,test_compiler,param1,swap_t_list_simulated[:1])
plt.plot(detuning_L,catch_result)
plt.show()


# if 1:
#     tl=np.linspace(0.1,10,100)
#     swap_t_list=[]
#     for i in range(8):
#         amp_list=phonon_rabi_measurement(tl,test_processor,test_compiler,swap_time_list=swap_t_list)
#         plt.plot(tl,amp_list)
#         t1_fitter=hbar_fitting.fitter(tl,np.array(amp_list))
#         swap_t=t1_fitter.fit_phonon_rabi()
#         swap_t_list.append(swap_t)

