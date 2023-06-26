# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:58:18 2023

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
    E1mag = np.sqrt(Gamma/(2*np.pi)) #2*np.pi*.05   
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
   
    wlas = Gamma*50 #THz
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
    
    Nt = (2**4)                                       #Number of Points
    time = T                                          #Length of time of tlist defined to be one period of the system
    dt = time/Nt                                      #Time point spacing in tlist
    tlist = np.linspace(0, time-dt, Nt)               #Combining everything to make tlist
    
    Ntau = (Nt)*150                           
    taume = (Ntau/Nt)*T                            
    dtau = taume/Ntau                              #time spacing in taulist - same as tlist!
    taulist = np.linspace(0, taume-dtau, Ntau)   

   
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
    
    Hf1  =  -(1/2)*np.array([[            0,Om1,0,0],
                             [            0,  0,0,0],
                             [            0,  0,0,0],
                             [            0,  0,0,0],]) \
        \
            -(1/2)*np.array([[ 0,0,            0,  0],
                             [ 0,0,            0,  0],
                             [ 0,0,            0,Om2],
                             [ 0,0,            0,  0]
                             ])
    
    Hb1 =   -(1/2)*np.array([[            0,   0,0,0],
                             [ -np.conj(Om1),   0,0,0],
                             [            0,   0,0,0],
                             [            0,   0,0,0]])\
        \
            -(1/2)*np.array([[0,0,           0,   0],
                             [0,0,           0,   0],
                             [0,0,           0,   0],
                             [0,0,-np.conj(Om2),   0]])
   
   
    Htot = Qobj(H_atom+Hf1+Hb1)#,
    
   

    lower_dot_1 = np.sqrt(Gamma)*Qobj(np.array([[0,1,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0]
                                 ]))
    lower_dot_2 = np.sqrt(Gamma)*Qobj(np.array([[0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,1],
                                 [0,0,0,0]
                                 ]))
    print('finished setting up the Hamiltonian')

    ############## Calculate the Emission Spectrum ###############################
    '''
    The Hamiltonian and collapse operators have been defined. Now, the first thing
    to do is supply some initial state rho_0
    Doing the ground state cause why not
    '''
    
    rho0 = basis(4,1)+basis(4,3)
    
    '''
    Next step is to iterate it FAR forward in time, i.e. far greater than any timescale present in the Hamiltonian
    
    Longest timescale in the Hamiltonian is Delta=.02277 Thz -> 5E-11 seconds
    
    Going to just use 1 second as this should be more than long enough. Might need 
    to multiply by the conversion factor as everything so far is in Thz...
    
    jk just gunna multiply the time scale by a big number
    
    It requires a tlist for some reason so I'll just take the last entry for the next stuff
    '''

 
    
    '''
    Next step is to iterate this steady state rho_s forward in time. I'll choose the times
    to be evenly spread out within T, the time scale of the Hamiltonian
    
    Also going through one time periods of the Hamiltonian so that I can graph the states
    and make sure I'm in the limit cycle
    '''
    TimeEvol =  mesolve(
            Htot,
            rho0,
            taulist,
            c_ops=[[lower_dot_1],[lower_dot_2]],
            options={"normalize_output": False},
            )

    states_array = np.stack([i.full() for i in TimeEvol.states])
 
    excite_pop_QD1 = states_array[:,1,1]**2
    excite_pop_QD2 = states_array[:,3,3]**2
 
    fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
    ax.plot(taulist/(2*np.pi/Gamma), excite_pop_QD1, color = 'b' )
    ax.plot(taulist/(2*np.pi/Gamma), excite_pop_QD2, color = 'r',linestyle='--' )
 
    ax.set_xlabel('$\\tau$ [$\Gamma$]')
    ax.set_ylabel("Amplitude") 
    ax.legend(['$QD_{1}$ resonance','$QD_{2}$ resonance'])
    plt.title('Fotso System Population Evolution')