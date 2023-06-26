# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:53:24 2023

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
   
    wlas = Gamma*140 #THz
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
     
                                                      #Taulist Definition
    periods = 1e+1
    Ntau =  int((Nt)*periods)                                 #50 times the number of points of tlist
    taume = (Ntau/Nt)*T                               #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau = taume/Ntau                                 #time spacing in taulist - same as tlist!
    taulist = np.linspace(0, taume-dtau, Ntau)        #Combining everything to make taulist, and I want taulist to end exactly at the beginning/end of a period to make some math easier on my end later
   
     
    Ntau2 = (Nt)*10000                              #50 times the number of points of tlist
    taume2 = (Ntau2/Nt)*T                             #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau2 = taume2/Ntau2                              #time spacing in taulist - same as tlist!
    taulist2 = np.linspace(0, taume2-dtau2, Ntau2)   
    if idz == 0:

        
        omega_array1 = np.fft.fftfreq(Ntau2,dtau)
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
    
        

    
    H_atom = ((wres1)/2)*np.array([[-1, 0, 0, 0],
                                   [ 0, 1, 0, 0],
                                   [ 0, 0, 0, 0],
                                   [ 0, 0, 0, 0]
                                   ]) \
        \
            +((wres2)/2)*np.array([[ 0, 0, 0, 0],
                                   [ 0, 0, 0, 0],
                                   [ 0, 0,-1, 0],
                                   [ 0, 0, 0, 1]
                                   ])
    
    Hf1  =  -(1/2)*np.array([[            0,Om1,0,0],
                             [0,  0,0,0],
                             [            0,  0,0,0],
                             [            0,  0,0,0],]) \
        \
            -(1/2)*np.array([[ 0,0,            0,  0],
                             [ 0,0,            0,  0],
                             [ 0,0,            0,Om2],
                             [ 0,0,0,  0]
                             ])
    
    Hb1 =   -(1/2)*np.array([[            0,0,0,0],
                             [ np.conj(Om1),   0,0,0],
                             [            0,   0,0,0],
                             [            0,   0,0,0]])\
        \
            -(1/2)*np.array([[0,0,           0,   0],
                             [0,0,           0,   0],
                             [0,0,           0,0],
                             [0,0,np.conj(Om2),   0]])
            
    
    
    H_pulse = (1/(0.2*Gamma))*np.array([[0,1,0,0],
                                        [1,0,0,0],
                                        [0,0,0,1],
                                        [0,0,1,0]
                                  ])
    freq_pulse = 0.2*Gamma
    N = 1000#int(periods*(freq_pulse/Gamma))
    string_time = []
    Hpulse_list = []
    for n in range(N):
        string_time+= [F'exp(-1j*({n}*l)*t)']
        
    string_time_test = '+'.join(string_time)
    Hpulse_list.append([Qobj(H_pulse), string_time_test])
        
        
    
    # H = [Qobj(H_atom),
    #       [Qobj(Hf1),'exp(1j * l * t )'],                    \
    #       [Qobj(Hb1), 'exp(-1j * l * t )']]    
    H = [Qobj(H_atom+Hf1+Hb1),
          *Hpulse_list]     
    
    Hargs = {'l':freq_pulse}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.


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

rho0 = basis(4,1)


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
timesense0 = 0
timesense1 = 1
timesense2 = 1e+5
opts = {"normalize_output": False,
        "atol":1e-6,
        "rtol":1e-8,
        "nsteps":1e+5}


TimeEvol_0 = flimesolve(
        Htot,
        rho0,
        taulist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense0,
        options=opts)
rhoss_0 = TimeEvol_0.states[-1]

TimeEvol_1 = flimesolve(
        Htot,
        rho0,
        taulist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense1,
        options=opts)
rhoss_1 = TimeEvol_1.states[-1]

TimeEvol_both = flimesolve(
        Htot,
        rho0,
        taulist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense2,
        options=opts)
rhoss_both = TimeEvol_both.states[-1]
'''
Next step is to iterate this steady state rho_s forward in time. I'll choose the times
to be evenly spread out within T, the time scale of the Hamiltonian

Also going through one time periods of the Hamiltonian so that I can graph the states
and make sure I'm in the limit cycle
'''


PeriodStates_0 = flimesolve(
        Htot,
        rhoss_0,
        taulist[-1]+tlist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense0,
        options=opts)

PeriodStates_1 = flimesolve(
        Htot,
        rhoss_1,
        taulist[-1]+tlist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense1,
        options=opts)

PeriodStates_both = flimesolve(
        Htot,
        rhoss_both,
        taulist[-1]+tlist,
        c_ops_and_rates = [[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
        T = T,
        args = Hargs,
        time_sense = timesense2,
        options=opts)


testg1_0 = np.zeros((len(tlist), len(taulist2)), dtype='complex_' ) 
testg1_1 = np.zeros((len(tlist), len(taulist2)), dtype='complex_' ) 
testg1_both = np.zeros((len(tlist), len(taulist2)), dtype='complex_' ) 
for tdx in range(len(tlist)):
    '''
    Start here tomorrow. You need to write taulist into the _make_solver
    arguments in Correlation, so that the FLiMESolver can construct
    properly. Then, since I'm probably dropping the automatic timer averaging,
    I'll need to use the for loop (for tdx in range(len(tlist)):) to calculate
    all the different g1s and then average them.'
    '''

    testg1_0[tdx] = correlation.correlation_2op_1t(Htot,
                                                  PeriodStates_0.states[tdx],
                                                  taulist = taulist[-1]+tlist[tdx]+taulist2,
                                                  c_ops=[[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
                                                  a_op = (lower_dot_1+lower_dot_2).dag(),
                                                  b_op = (lower_dot_1+lower_dot_2),
                                                  solver="fme",
                                                  reverse = True,
                                                  options = {'T':T,'time sense':timesense0},
                                                  args = Hargs)[0]
    
    testg1_1[tdx] = correlation.correlation_2op_1t(Htot,
                                                  PeriodStates_1.states[tdx],
                                                  taulist = taulist[-1]+tlist[tdx]+taulist2,
                                                  c_ops=[[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
                                                  a_op = (lower_dot_1+lower_dot_2).dag(),
                                                  b_op = (lower_dot_1+lower_dot_2),
                                                  solver="fme",
                                                  reverse = True,
                                                  options = {'T':T,'time sense':timesense1},
                                                  args = Hargs)[0]
    
    testg1_both[tdx] = correlation.correlation_2op_1t(Htot,
                                                  PeriodStates_both.states[tdx],
                                                  taulist = taulist[-1]+tlist[tdx]+taulist2,
                                                  c_ops=[[lower_dot_1,Gamma],[lower_dot_2,Gamma]],
                                                  a_op = (lower_dot_1+lower_dot_2).dag(),
                                                  b_op = (lower_dot_1+lower_dot_2),
                                                  solver="fme",
                                                  reverse = True,
                                                  options = {'T':T,'time sense':timesense2},
                                                  args = Hargs)[0]

g1_0_avg = np.average(testg1_0,axis=0)
specF_0 = np.fft.fft(g1_0_avg,axis=0)
specF_0 = np.fft.fftshift(specF_0)/len(g1_0_avg)
Z_0 = specF_0

g1_1_avg = np.average(testg1_1,axis=0)
specF_1 = np.fft.fft(g1_1_avg,axis=0)
specF_1 = np.fft.fftshift(specF_1)/len(g1_1_avg)
Z_1 = specF_1

g1_both_avg = np.average(testg1_both,axis=0)
specF_both = np.fft.fft(g1_both_avg,axis=0)
specF_both = np.fft.fftshift(specF_both)/len(g1_both_avg)
Z_both = specF_both
    
freqlims = [-30,30]


frequency_range = (((omega_array+(wlas/(2*np.pi)))/(Gamma/(2*np.pi))))
idx0 = np.where(abs(frequency_range-freqlims[0]) == np.amin(abs((frequency_range-freqlims[0] ))))[0][0]
idxf = np.where(abs(frequency_range-freqlims[1]) == np.amin(abs((frequency_range-freqlims[1] ))))[0][0]

plot_freq_range = frequency_range[idx0:idxf]
Z0_truncated = np.stack(Z_0[idx0:idxf])
Z1_truncated = np.stack(Z_1[idx0:idxf])
Zboth_truncated = np.stack(Z_both[idx0:idxf])

fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
# ax.semilogy( (plot_freq_range ), ZF_truncated, color = 'r' )
ax.semilogy( (plot_freq_range), Z0_truncated, color = 'r' )
ax.semilogy( (plot_freq_range), Z1_truncated, color = 'g', linewidth = 3 )
ax.semilogy( (plot_freq_range), Zboth_truncated, color = 'k' , alpha= 0.8)
# ax.axvline(x=(-(wres1-wlas)/(Gamma)), color='r', linestyle = 'dashed')
# ax.axvline(x=(-(wres2-wlas)/(Gamma)), color='g', linestyle = 'dashed')
ax.set_xlabel('Detuning [$\Gamma$]')
ax.set_ylabel("Amplitude") 
ax.legend(['time_sense = 0','time_sense = 1$','time_sense = 1e+5'])


# fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
# ax.semilogy(((omega_array+(wlas/(2*np.pi)))/(Gamma/(2*np.pi))), ZF, color = 'r' )
# ax.axvline(x=(-(wres1-wlas)/(Gamma)), color='r', linestyle = 'dashed')
# ax.axvline(x=(-(wres2-wlas)/(Gamma)), color='g', linestyle = 'dashed')
# ax.set_xlabel('Detuning [THz]')
# ax.set_ylabel("Amplitude") 
# ax.set_xlabel('Detuning [$\Gamma$]')
# ax.set_ylabel("Amplitude") 
# ax.legend(['Emission','$QD_{1}$ resonance','$QD_{2}$ resonance'])