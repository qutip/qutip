# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:35:37 2023

@author: Fenton
"""

import matplotlib.pyplot as plt
# import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, 'C:/Users/Fenton/Documents/GitHub/qutip')
from qutip import Qobj,basis,mesolve
import numpy as np
from qutip import *
from qutip.ui.progressbar import BaseProgressBar
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib


from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')

'''
Modelling on Josepheson junction QUbits, because I see no reason not to
Google says these have transition frequencies in the 1-10 GHz range.

Choosing an intermediate value of 5.
'''
wres = 2*np.pi*280 #THz




  

'''
For loop used to create the range of powers for laser 2.
The way it's set up now, the Om_2 will be integers that go up to the value of power_range
'''

Tau_array = [10,100,1000,10000,50000,100000,200000]

quicksolve_state_array = []
flimesolve_state_array = []
mesolve_state_array = []

quicksolve_time = []
flimesolve_time = []
mesolve_time = []

for idz, periods in enumerate(Tau_array):
    print('Working on Spectra number', idz, 'of',len(Tau_array))
    ############## Experimentally adjustable parameters #####################
    #electric field definitions
    
    '''
    In THz so .01 is rather large
    
    Defining the magnitude off of the resonance frequency because 
        the paper I'm trying to reproduce gives coupling in terms of the 
        resonant frequency. Since my dipole moment has magnitude 1, I define
        the coupling constant here, effectively.'
    '''
    E1mag = wres*0.5
   
    '''
    Defining the polarization that will dot with the dipole moment to form the
    Rabi Frequencies
    '''
    E1pol = np.sqrt(1/2)*np.array([0, 1, 0]); 
   
    
    '''
    Total E field
    '''
    E1 = E1mag*E1pol
   
   
    ############## Hamiltonian parameters ###################################
    '''
    Going to define the states for clarity, although I'll be using the "mat" function
        to make any matrix elements
    
    One ground state, one excited state
    '''
    Hdim = 2 # dimension of Hilbert space
    gnd0 = basis(2,0)       #|0>
    gnd1 = basis(2,1)       #|1>
    
    '''
    Defining the function to make matrix operators |i><j|
    '''
    def mat(i,j):
        return(basis(2,i)*basis(2,j).dag())
    

    '''
    Defining the Dipole moment of the QD that will dot with the laser polarization
        to form the Rabi Frequency and Rabi Frequency Tilde
    '''
    dmag = 1
    d = dmag *  np.sqrt(1/2) * np.array([1, -1j, 0]) 
    
    Om1  = np.dot(d,        E1)
    Om1t = np.dot(d,np.conj(E1))
    
   
    wlas = wres
   
    T = 2*np.pi/abs(wlas) # period of the Hamiltonian
    Hargs = {'l': (wlas)}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.
    w = Hargs['l']
    
    '''
    Defining the spontaneous emission operators that will link the two states
    Another paper I've looked up seems to say that decay values of ~275 kHz
        are common. I think.
    '''
    Gamma = 2 * np.pi * 2.5e-3   #in THz, roughly equivalent to 1 micro eV
    spont_emis = np.sqrt(Gamma) * mat(0,1)           # Spontaneous emission operator   
    

      
        
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
     
                                                      #Taulist Definition
    Ntau =  (Nt)*periods                                 #50 times the number of points of tlist
    taume = (Ntau/Nt)*T                               #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau = taume/Ntau                                 #time spacing in taulist - same as tlist!
    taulist = np.linspace(0, taume-dtau, Ntau)        #Combining everything to make taulist, and I want taulist to end exactly at the beginning/end of a period to make some math easier on my end later
   
  
    ################################# Hamiltonian #################################
    '''
    Finally, I'll define the full system Hamiltonian. Due to the manner in which
        QuTiP accepts Hamiltonians, the Hamiltonian must be defined in separate terms
        based on time dependence. Due to that, the Hamiltonian will have three parts.
        One time independent part, one part that rotates forwards in time, and one
        part that rotates backwards in time. For full derivation, look at my "2LS
        Bichromatic Excitation" subtab under my 4/13/21 report
    '''
   
    
    H_atom = (wres/2)*np.array([[-1,0],
                                [ 0,1]])
    
    Hf1  = (1/2)*np.array([[    0,Om1],
                                [np.conj(Om1t),  0]])
    
    Hb1 = (1/2)*np.array([[   0,Om1t],
                                [np.conj(Om1),   0]])
    
  
    
    H0 =  Qobj(H_atom)                                  #Time independant Term
    Hf1 =  Qobj(Hf1)                     #Forward Rotating Term
    Hb1 =  Qobj(Hb1)                     #Backward Rotating Term
  
    Htot= [H0,                                        \
        [Hf1,'exp(1j * l * t )'],                    \
        [Hb1, 'exp(-1j * l * t )']]                                        #Full Hamiltonian in string format, a form acceptable to QuTiP
    
   
    
    print('finished setting up the Hamiltonian')

    ############## Calculate the Emission Spectrum ###############################
    '''
    The Hamiltonian and collapse operators have been defined. Now, the first thing
    to do is supply some initial state rho_0
    Doing the ground state cause why not
    '''
    
    rho0 = basis(2,0)
    
    '''
    Next step is to iterate it FAR forward in time, i.e. far greater than any timescale present in the Hamiltonian
    
    Longest timescale in the Hamiltonian is Delta=.02277 Thz -> 5E-11 seconds
    
    Going to just use 1 second as this should be more than long enough. Might need 
    to multiply by the conversion factor as everything so far is in Thz...
    
    jk just gunna multiply the time scale by a big number
    
    It requires a tlist for some reason so I'll just take the last entry for the next stuff
    '''
    QuickSolve = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=0,
        quicksolve = True,
        options={"normalize_output": False})
    quicksolve_state_array.append(QuickSolve.states[-1].full())
    quicksolve_time.append(QuickSolve.stats["run time"])

    FlimeSolve = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=0,
        quicksolve = False,
        options={"normalize_output": False})
    flimesolve_state_array.append(FlimeSolve.states[-1].full())
    flimesolve_time.append(FlimeSolve.stats["run time"])
   
    MESolve = mesolve(
        Htot,
        rho0,
        taulist,
        c_ops=[np.sqrt(Gamma)*destroy(2)],
        options={"normalize_output": False},
        args=Hargs,
        )
    mesolve_state_array.append(MESolve.states[-1].full())
    mesolve_time.append(MESolve.stats["run time"])
    
quicksolvestates = np.stack([i for i in quicksolve_state_array])
flimestates = np.stack([i for i in flimesolve_state_array])
mestates = np.stack([i for i in mesolve_state_array])

quicksolvestates_full = np.stack([i.full() for i in QuickSolve.states])
flimestates_full = np.stack([i.full() for i in FlimeSolve.states])
mestates_full = np.stack([i.full() for i in MESolve.states])


quicksolve_time = np.stack(quicksolve_time)
flimesolve_time = np.stack(flimesolve_time)
mesolve_time = np.stack(mesolve_time)


fig, ax = plt.subplots(3,1)                                                    

ax[0].plot(taulist/T,np.sqrt((mestates_full[:,1,1])**2), color = 'black')
ax[0].plot(taulist/T,np.sqrt((quicksolvestates_full[:,1,1])**2), color = 'green', linestyle = ':' )
ax[0].plot(taulist/T,np.sqrt((flimestates_full[:,1,1])**2), color = 'blue' ,alpha = 0.6)
ax[0].legend(['MEsolve','QuickSolve','FLiMESolve'])
ax[0].set_ylabel('Excited State Population')  

# ax[1].plot(taulist/T,np.sqrt((mestates_full[:,1,1]-flimestates_full[:,1,1])**2), color = 'blue' )
# ax[1].plot(taulist/T,np.sqrt((mestates_full[:,1,1]-quicksolvestates_full[:,1,1])**2), color = 'green',linestyle=':' )
# ax[1].legend(['FLiMESolve','QuickSolve'])
# ax[1].set_ylabel('RMS difference from MEsolve')  
# ax[1].set_xlabel('Periods of evolution ($\\tau$)')

# ax[2].scatter(Tau_array,flimesolve_time/mesolve_time, color = 'blue' )
# ax[2].scatter(Tau_array,mesolve_time/mesolve_time, color = 'black')
# ax[2].scatter(Tau_array,quicksolve_time/mesolve_time, color = 'red' )
# ax[2].legend(['FLiMESolve','MEsolve','QuickSolve'])
# ax[2].set_ylabel('Solution time Divided by MESolve time')  
# ax[2].set_xlabel('Periods of Evolution')


# print('QuickSolve runtime =',QuickSolve.stats["run time"])
# print('FLiMESolve runtime =',FlimeSolve.stats["run time"])
# print('MESolve runtime =',MESolve.stats["run time"])
