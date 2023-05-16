# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:36:06 2023

@author: Fenton
"""


import matplotlib.pyplot as plt
from qutip import flimesolve,Qobj,basis,mesolve
import numpy as np
from qutip import *
from qutip.ui.progressbar import BaseProgressBar
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

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

Tau_array = [1000]

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
    
    Nt = (2**2)                                       #Number of Points
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
    MESolve = mesolve(
        Htot,
        rho0,
        taulist,
        c_ops=[np.sqrt(Gamma)*destroy(2)],
        options={"normalize_output": False},
        args=Hargs,
        )
    mesolve_time = (MESolve.stats["run time"])
    

    FlimeSolve0 = flimesolve(
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
    FlimeSolve0t = (FlimeSolve0.stats["run time"])/mesolve_time
    
    FlimeSolve1 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=1,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve1t = (FlimeSolve1.stats["run time"])/mesolve_time
     
    FlimeSolve2 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=2,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve2t = (FlimeSolve2.stats["run time"])/mesolve_time
     
    FlimeSolve3 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=3,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve3t = (FlimeSolve3.stats["run time"])/mesolve_time
     
    FlimeSolve4 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=4,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve4t = (FlimeSolve4.stats["run time"])/mesolve_time
     
    FlimeSolve5 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=5,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve5t = (FlimeSolve5.stats["run time"])/mesolve_time
     
    FlimeSolve6 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=6,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve6t = (FlimeSolve6.stats["run time"])/mesolve_time
     
    FlimeSolve7 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=7,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve7t = (FlimeSolve7.stats["run time"])/mesolve_time
     
    FlimeSolve8 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=8,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve8t = (FlimeSolve8.stats["run time"])/mesolve_time
      
    FlimeSolve9 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=9,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve9t = (FlimeSolve9.stats["run time"])/mesolve_time
      
    FlimeSolve10 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[Gamma],
        T=T,
        args=Hargs,
        time_sense=10,
        quicksolve = False,
        options={"normalize_output": False})
    FlimeSolve10t = (FlimeSolve10.stats["run time"])/mesolve_time
      
 


time_quotients = [FlimeSolve0t,FlimeSolve1t,FlimeSolve2t,FlimeSolve3t,FlimeSolve4t,FlimeSolve5t,FlimeSolve6t,FlimeSolve7t,FlimeSolve8t,FlimeSolve9t,FlimeSolve10t]
time_vals = ['0','1','2','3','4','5','6','7','8','9','10']



fig, ax = plt.subplots(1,1)                                                    

ax.scatter(time_vals,time_quotients)
# ax.legend(['0 $\omega$','0.1 $\omega$','0.2 $\omega$','0.3 $\omega$','0.4 $\omega$','0.5 $\omega$','1 $\omega$','10 $\omega$','20 $\omega$','50 $\omega$','100 $\omega$'])
ax.plot(1,linestyle='dashed')
ax.set_ylabel('solution time quotients versus MESolve')  
ax.set_xlabel('time dependancy of FLiMESolve')
