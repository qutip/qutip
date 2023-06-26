# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:30:49 2023

@author: Fenton
"""


import matplotlib.pyplot as plt
from qutip import flimesolve,mesolve,Qobj,basis,destroy,correlation,FloquetBasis
import numpy as np
# from qutip import *
# from qutip.ui.progressbar import BaseProgressBar
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib


from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')



'''
For loop used to create the range of powers for laser 2.
The way it's set up now, the Om_2 will be integers that go up to the value of power_range
'''





# detuning_array = np.linspace(-0.5,0.5,101)
detuning_array = [0.0]
for idz, detuning in enumerate(detuning_array):
    print('Working on Spectra number', idz+1, 'of',len(detuning_array))
    ############## Experimentally adjustable parameters #####################
    #electric field definitions
    Gamma = 2*2*np.pi    #in THz, roughly equivalent to 1 micro eV
    '''
    In THz so .01 is rather large
    
    Defining the magnitude off of the resonance frequency because 
        the paper I'm trying to reproduce gives coupling in terms of the 
        resonant frequency. Since my dipole moment has magnitude 1, I define
        the coupling constant here, effectively.'
    '''
    E1mag = Gamma/2 #2*np.pi*.05   
    '''
    Defining the polarization that will dot with the dipole moment to form the
    Rabi Frequencies
    '''
    E1pol = 1 
   
    
    '''
    Total E field
    '''
    E1 = E1mag*E1pol
   
    ############## Hamiltonian parameters ###################################
 
    '''
    Defining the Dipole moment of the QD that will dot with the laser polarization
        to form the Rabi Frequency and Rabi Frequency Tilde
    '''
    dmag = 1
    d1 = 1 
    d2 = 1

    Om1  = np.dot(d1,        E1)
    Om1t = np.dot(d1,np.conj(E1))
    
    Om2  = np.dot(d2,        E1)
    Om2t = np.dot(d2,np.conj(E1))
   
    wlas = Gamma*100 #THz
    wres1 = wlas+3*Gamma
    wres2 = wlas-4*Gamma

   
    T = 2*np.pi/wlas # period of the Hamiltonian
    
    
    '''
    Defining the spontaneous emission operators that will link the two states
    Another paper I've looked up seems to say that decay values of ~275 kHz
        are common. I think.
    '''
    

      
        
    '''
    The following tlist definitions are for different parts of the following calulations
    The first is to iterate the dnesity matrix to some time far in the future, and seems to need a lot of steps for some reason
    The second dictates the number of t values evenly distributed over the "limit cycle" that will be averaged over later
    The third is for the tau values that are used to iterate the matrix forward after multiplying by the B operator
    '''
    
    Nt = (2**2)                                       #Number of Points
    time = T                                          #Length of time of tlist defined to be one period of the system
    dt = time/Nt                                      #Time point spacing in tlist
    tlist = np.linspace(0, time-dt, Nt)               #Combining everything to make tlist

    Ntau2 = (Nt)*300                            
    taume2 = (Ntau2/Nt)*T                            
    dtau2 = taume2/Ntau2                              #time spacing in taulist - same as tlist!
    taulist2 = np.linspace(0, taume2-dtau2, Ntau2)   
    if idz == 0:

        
        omega_array1 = np.fft.fftfreq(Ntau2,dtau2)
        omega_array = np.fft.fftshift(omega_array1)
        
        ZF = np.zeros( (len(detuning_array), len(omega_array1)) )
   
    ################################# Hamiltonian #################################
    '''
    Finally, I'll define the full system Hamiltonian. Due to the manner in which
        QuTiP accepts Hamiltonians, the Hamiltonian must be defined in separate terms
        based on time dependence. Due to that, the Hamiltonian will have three parts.
        One time independent part, one part that rotates forwards in time, and one
        part that rotates backwards in time. For full derivation, look at my "2LS
        Bichromatic Excitation" subtab under my 4/13/21 report
    '''
    
        

    
    H_atom = ((wres1-wlas)/2)*np.array([[-1, 0, 0, 0],
                                   [ 0, 1, 0, 0],
                                   [ 0, 0, 0, 0],
                                   [ 0, 0, 0, 0]
                                   ]) \
        \
            +((wres2-wlas)/2)*np.array([[ 0, 0, 0, 0],
                                   [ 0, 0, 0, 0],
                                   [ 0, 0,-1, 0],
                                   [ 0, 0, 0, 1]
                                   ])
    
    Hf1  =  -(1j/2)*np.array([[            0,Om1,0,0],
                             [            0,  0,0,0],
                             [            0,  0,0,0],
                             [            0,  0,0,0],]) \
        \
            -(1j/2)*np.array([[ 0,0,            0,  0],
                             [ 0,0,            0,  0],
                             [ 0,0,            0,Om2],
                             [ 0,0,            0,  0]
                             ])
    
    Hb1 =   -(1j/2)*np.array([[            0,   0,0,0],
                             [ -np.conj(Om1),   0,0,0],
                             [            0,   0,0,0],
                             [            0,   0,0,0]])\
        \
            -(1j/2)*np.array([[0,0,           0,   0],
                             [0,0,           0,   0],
                             [0,0,           0,   0],
                             [0,0,-np.conj(Om2),   0]])
            
    
    
    H_pulse = (0/(0.2*Gamma))*np.array([[0,1,0,0],
                                        [1,0,0,0],
                                        [0,0,0,1],
                                        [0,0,1,0]
                                  ])
    freq_pulse = Gamma
    N = 10#int(periods*(freq_pulse/Gamma))
    string_time = []
    Hpulse_list = []
    for n in range(N):
        string_time+= [F'exp(-1j*({n}*0.2*{freq_pulse}-l)*t)']
        
    string_time_test = '+'.join(string_time)
    Hpulse_list.append([Qobj(H_pulse), string_time_test])
        
        
    
    # H = [Qobj(H_atom),
          # [Qobj(Hf1),'exp(1j * l * t )'],                    \
          # [Qobj(Hb1), 'exp(-1j * l * t )']]    
    H = [Qobj(H_atom+Hf1+Hb1),
          *Hpulse_list]     
    
    Hargs = {'l':wlas}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.


    lower_dot_1 = Qobj(np.array([[0,1,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0]
                                 ]))
    lower_dot_2 = Qobj(np.array([[0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,1],
                                 [0,0,0,0]
                                 ]))
    print('finished setting up the Hamiltonian')

#########77777777##### Calculate the Emission Spectrum ###############################
'''
The Hamiltonian and collapse operators have been defined. Now, the first thing
to do is supply some initial state rho_0
Doing the ground state cause why not
'''



rho0 = basis(4,1)+basis(4,3)

'''
Creating floquet basis object so it doesn't need to be solved multiple times'
'''

Htot = FloquetBasis(H, T,args=Hargs)

'''
Next step is to iterate it FAR forward in time, i.e. far greater than any timescale present in the Hamiltonian

Longest timescale in the Hamiltonian is Delta=.02277 Thz -> 5E-11 seconds

Going to just use 1 second as this should be more than long enough. Might need 
to multiply by the conversion factor as everything so far is in Thz...

jk just gunna multiply the time scale by a big number

It requires a tlist for some reason so I'll just take the last entry for the next stuff
'''
#Common settings
timesense = 0
opts = {"normalize_output": False,
        "atol":1e-6,
        "rtol":1e-8,
        "nsteps":1e+5}

'''
Next step is to iterate this steady state rho_s forward in time. I'll choose the times
to be evenly spread out within T, the time scale of the Hamiltonian

Also going through one time periods of the Hamiltonian so that I can graph the states
and make sure I'm in the limit cycle
'''




PeriodStates_both = flimesolve(
        Htot,
        rho0,
        tlist,
        c_ops_and_rates = [[lower_dot_1,np.sqrt(Gamma)],[lower_dot_2,np.sqrt(Gamma)]],
        T = T,
        args = Hargs,
        time_sense = timesense,
        options=opts)
periodstates=PeriodStates_both.states
'''
The next step is to create the g2 operater multiplication. Given by...I think
Mark Fox? The operator is
<A(t0)B(t1)C(t1)D(t0)> == Tr<B*C*V(t1,t0){D*p(t0)*A}>

where in this case
A = a.dag()
D = a
B = a4.dag()
C = a4
'''
'''
The regular g2 function from qutip wont work here, as I have to mix
QD operators in the G2
'''

'''
Defining the operators a3, a4 how Fotso does
'''
a3 = Qobj(np.sqrt(1/2)*(lower_dot_1.full()+1j*lower_dot_2.full()))
a4 = Qobj(np.sqrt(1/2)*(1j*lower_dot_1.full()+lower_dot_2.full()))

# DAstates = [a3 * state * a3.dag() for state in periodstates]
DAstates = [a3 * periodstates[0] * a3.dag()]
state_norms_a3_unavg = []
state_norms_a4_unavg = []
states_unavg =[]
for tdx in range(len(DAstates)):
    print('working on', tdx+1, 'of', len(DAstates))
    DAstates_evol_both = flimesolve(Htot, 
                                    DAstates[tdx], 
                                    taulist = tlist[tdx]+taulist2,
                                    c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
                                    T = T,
                                    args = Hargs,
                                    time_sense = timesense,
                                    options=opts)
    
    PeriodStates_evol = flimesolve(Htot, 
                                    PeriodStates_both.states[tdx], 
                                    taulist = tlist[tdx]+taulist2,
                                    c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
                                    T = T,
                                    args = Hargs,
                                    time_sense = timesense,
                                    options=opts)

    statesRow = []
    state_norms_row_a3 = []
    state_norms_row_a4 = []
    for idz, state in enumerate(DAstates_evol_both.states):
        statesRow.append((a4.dag()*a4*state).full())
        state_norms_row_a3.append((a3.dag()*a3*PeriodStates_evol.states[idz]).full())
        state_norms_row_a4.append((a4.dag()*a4*PeriodStates_evol.states[idz]).full())
        
    states_unavg.append(np.stack(statesRow)  )    
    state_norms_a3_unavg.append(np.stack(state_norms_row_a3)  )
    state_norms_a4_unavg.append(np.stack(state_norms_row_a4)  )


g2_num_avg = np.average(states_unavg,axis=0)
g2_a3_denom_avg = np.average(state_norms_a3_unavg,axis=0)
g2_a4_denom_avg = np.average(state_norms_a4_unavg,axis=0)

g2_num_trace =  np.trace(g2_num_avg,axis1=1,axis2=2)
g2_a3_denom_trace =  np.trace(g2_a3_denom_avg,axis1=1,axis2=2)
g2_a4_denom_trace =  np.trace(g2_a4_denom_avg,axis1=1,axis2=2)

g2 = g2_num_trace/(g2_a3_denom_trace[0]*g2_a4_denom_trace)


fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
ax.plot(taulist2, g2, color = 'b' )
ax.axhline(y=1, color='k', linestyle = 'dashed')

ax.set_xlabel('$\\tau$ [$\Gamma$]')
ax.set_ylabel("Amplitude") 
ax.legend(['Emission','$QD_{1}$ resonance','$QD_{2}$ resonance'])
plt.title(F'Fotso System no Control protocol time_sense = {timesense}')
    
# freqlims = [-30,30]


# frequency_range = (((omega_array+(wlas/(2*np.pi)))/(Gamma/(2*np.pi))))
# idx0 = np.where(abs(frequency_range-freqlims[0]) == np.amin(abs((frequency_range-freqlims[0] ))))[0][0]
# idxf = np.where(abs(frequency_range-freqlims[1]) == np.amin(abs((frequency_range-freqlims[1] ))))[0][0]

# plot_freq_range = frequency_range[idx0:idxf]
# Z0_truncated = np.stack(Z_0[idx0:idxf])
# Z1_truncated = np.stack(Z_1[idx0:idxf])
# Zboth_truncated = np.stack(Z_both[idx0:idxf])

# fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
# # ax.semilogy( (plot_freq_range ), ZF_truncated, color = 'r' )
# ax.semilogy( (plot_freq_range), Z0_truncated, color = 'r' )
# ax.semilogy( (plot_freq_range), Z1_truncated, color = 'g' )
# ax.semilogy( (plot_freq_range), Zboth_truncated, color = 'k' )
# # ax.axvline(x=(-(wres1-wlas)/(Gamma)), color='r', linestyle = 'dashed')
# # ax.axvline(x=(-(wres2-wlas)/(Gamma)), color='g', linestyle = 'dashed')
# ax.set_xlabel('Detuning [$\Gamma$]')
# ax.set_ylabel("Amplitude") 
# ax.legend(['$QD_{1}$','$QD_{2}$','Summed emission'])

