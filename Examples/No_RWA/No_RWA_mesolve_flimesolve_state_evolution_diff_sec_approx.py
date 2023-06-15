# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:27:36 2023

@author: Fenton
"""

import matplotlib.pyplot as plt
from qutip import flimesolve,mesolve,Qobj,basis,destroy,correlation,sigmaz
import numpy as np
# from qutip import *
# from qutip.ui.progressbar import BaseProgressBar
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib


from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')

'''
Modelling on Josepheson junction QUbits, because I see no reason not to
Google says these have transition frequencies in the 1-10 GHz range.

Choosing an intermediate value of 5.
'''
wres = 2*np.pi*50 #THz

'''
For loop used to create the range of powers for laser 2.
The way it's set up now, the Om_2 will be integers that go up to the value of power_range
'''






period_array = [50]#np.linspace(1,50,100)
fsolve_time_array0 = []
fsolve_time_array1 = []
fsolve_time_array10 = []
fsolve_time_array100 = []
fsolve_time_array1000 = []
msolve_time_array = []
for idz, periods in enumerate(period_array):
    print('Working on Spectra number', idz+1, 'of',len(period_array))
    ############## Experimentally adjustable parameters #####################
    #electric field definitions
    
    '''
    In THz so .01 is rather large
    
    Defining the magnitude off of the resonance frequency because 
        the paper I'm trying to reproduce gives coupling in terms of the 
        resonant frequency. Since my dipole moment has magnitude 1, I define
        the coupling constant here, effectively.'
    '''
    a = 1
    E1mag = wres*a/2   
    '''
    Defining the polarization that will dot with the dipole moment to form the
    Rabi Frequencies
    '''
    E1pol = np.sqrt(1/2)*np.array([1, 1, 0]); 
   
    
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
    d = dmag *  np.sqrt(1/2) * np.array([1, 1, 0]) 
    
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
    Gamma = wres*0.03   #in THz, roughly equivalent to 1 micro eV
    spont_emis = np.sqrt(Gamma) * mat(0,1)           # Spontaneous emission operator   
    
    # dep_op = sigmaz()
    # dep_rate = wres*0.06
        
    '''
    The following tlist definitions are for different parts of the following calulations
    The first is to iterate the dnesity matrix to some time far in the future, and seems to need a lot of steps for some reason
    The second dictates the number of t values evenly distributed over the "limit cycle" that will be averaged over later
    The third is for the tau values that are used to iterate the matrix forward after multiplying by the B operator
    '''
    
    Nt = (2**8)                                       #Number of Points
    time = T                                          #Length of time of tlist defined to be one period of the system
    dt = time/Nt                                      #Time point spacing in tlist
    tlist = np.linspace(0, time-dt, Nt)               #Combining everything to make tlist
     
    Ntau =  int((Nt)*periods)                                 #50 times the number of points of tlist
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
    
    Hf1  = -(1/2)*np.array([[    0,Om1],
                                [np.conj(Om1t),  0]])
    
    Hb1 = -(1/2)*np.array([[   0,Om1t],
                                [np.conj(Om1),   0]])
    
  
    
    H0 =  Qobj(H_atom)                                  #Time independant Term
    Hf1 =  Qobj(Hf1)                     #Forward Rotating Term
    Hb1 =  Qobj(Hb1)                     #Backward Rotating Term
  
    H= [H0,                                        \
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

    
    TimeEvolF0 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 0,
            quicksolve = True,
            options={"normalize_output": False,
                      'atol':1e-12,
                      'rtol':1e-8})
    fsolve_time_array0.append(TimeEvolF0.stats["run time"])
    
    TimeEvolF1 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 0,
            quicksolve = False,
            options={"normalize_output": False,
                      'atol':1e-12,
                      'rtol':1e-8})
    fsolve_time_array1.append(TimeEvolF1.stats["run time"])
    
    TimeEvolF10 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 1e+2,
            quicksolve = False,
            options={"normalize_output": False,
                      'atol':1e-12,
                      'rtol':1e-8})
    fsolve_time_array10.append(TimeEvolF10.stats["run time"])
    
    TimeEvolF100 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 1e+3,
            quicksolve = False,
            options={"normalize_output": False,
                      'atol':1e-12,
                      'rtol':1e-8})
    fsolve_time_array100.append(TimeEvolF100.stats["run time"])
    
    TimeEvolF1000 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 1e+8,
            quicksolve = False,
            options={"normalize_output": False,
                     'atol':1e-12,
                     'rtol':1e-8})
    fsolve_time_array1000.append(TimeEvolF1000.stats["run time"])

    TimeEvolM =  mesolve(
            H,
            rho0,
            taulist,
            c_ops=[np.sqrt(Gamma)*destroy(2)],
            options={"normalize_output": False,
                     'atol':1e-8,
                     'rtol':1e-6},
            args=Hargs,
            )
    msolve_time_array.append(TimeEvolM.stats["run time"])
 
    
    '''
    Next step is to iterate this steady state rho_s forward in time. I'll choose the times
    to be evenly spread out within T, the time scale of the Hamiltonian
    
    Also going through one time periods of the Hamiltonian so that I can graph the states
    and make sure I'm in the limit cycle
    '''


 
 
fstates0 = np.array([i.full() for i in TimeEvolF0.states])
fstates1 = np.array([i.full() for i in TimeEvolF1.states])
fstates10 = np.array([i.full() for i in TimeEvolF10.states])
fstates100 = np.array([i.full() for i in TimeEvolF100.states])
fstates1000 = np.array([i.full() for i in TimeEvolF1000.states])
mstates = np.array([i.full() for i in TimeEvolM.states])

fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
ax.plot(  taulist/T,np.sqrt(fstates0[:,1,1]**2), color = 'red')
ax.plot(  taulist/T,np.sqrt(fstates1[:,1,1]**2), color = 'orange')
ax.plot(  taulist/T,np.sqrt(fstates10[:,1,1]**2), color = 'yellow')
ax.plot(  taulist/T,np.sqrt(fstates100[:,1,1]**2), color = 'green')
ax.plot(  taulist/T,np.sqrt(fstates1000[:,1,1]**2), color = 'blue')
ax.plot(  taulist/T,np.sqrt(mstates[:,1,1]**2), color = 'black')
ax.legend(['FLiME quicksolve','FLiME time_sense=0','FLiME time_sense=1e+2','FLiME time_sense=1e+3','FLiME time_sense=1e+8','MESolve'])
ax.set_xlabel('Evolution time (t/$\\tau$)')
ax.set_ylabel('Excited State Population')

perc_dev0 = np.sqrt(np.average(((mstates[1::,1,1]-fstates0[1::,1,1])/mstates[1::,1,1])**2))
perc_dev1 = np.sqrt(np.average(((mstates[1::,1,1]-fstates1[1::,1,1])/mstates[1::,1,1])**2))
perc_dev10 = np.sqrt(np.average(((mstates[1::,1,1]-fstates10[1::,1,1])/mstates[1::,1,1])**2))
perc_dev100 = np.sqrt(np.average(((mstates[1::,1,1]-fstates100[1::,1,1])/mstates[1::,1,1])**2))
perc_dev1000 = np.sqrt(np.average(((mstates[1::,1,1]-fstates1000[1::,1,1])/mstates[1::,1,1])**2))


# perc_dev0 = np.sqrt((((mstates[1::,1,1]-fstates0[1::,1,1])/mstates[1::,1,1])**2))
# perc_dev1 = np.sqrt((((mstates[1::,1,1]-fstates1[1::,1,1])/mstates[1::,1,1])**2))
# perc_dev10 = np.sqrt((((mstates[1::,1,1]-fstates10[1::,1,1])/mstates[1::,1,1])**2))
# perc_dev100 = np.sqrt((((mstates[1::,1,1]-fstates100[1::,1,1])/mstates[1::,1,1])**2))
# perc_dev1000 = np.sqrt((((mstates[1::,1,1]-fstates1000[1::,1,1])/mstates[1::,1,1])**2))

# fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
# ax.plot(  taulist[1::]/T,perc_dev0, color = 'black')
# ax.plot(  taulist[1::]/T,perc_dev1, color = 'magenta')
# ax.plot(  taulist[1::]/T,perc_dev10, color = 'blue')
# ax.plot(  taulist[1::]/T,perc_dev100, color = 'green')
# ax.plot(  taulist[1::]/T,perc_dev1000, color = 'yellow')
# ax.set_ylabel('Percent deviation')
# ax.set_xlabel('Evolution time (t/$\\tau$)')


# time_quotient0 = np.array([fsolve_time_array0[i]/msolve_time_array[i] for i in range(len(period_array))])
# time_quotient1 = np.array([fsolve_time_array1[i]/msolve_time_array[i] for i in range(len(period_array))])
# time_quotient10 = np.array([fsolve_time_array10[i]/msolve_time_array[i] for i in range(len(period_array))])
# time_quotient100 = np.array([fsolve_time_array100[i]/msolve_time_array[i] for i in range(len(period_array))])
# time_quotient1000 = np.array([fsolve_time_array1000[i]/msolve_time_array[i] for i in range(len(period_array))])

# fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
# ax.scatter(  period_array,time_quotient0, color = 'black')
# ax.scatter(  period_array,time_quotient1, color = 'magenta')
# ax.scatter(  period_array,time_quotient10, color = 'blue')
# ax.scatter(  period_array,time_quotient100, color = 'green')
# ax.scatter(  period_array,time_quotient1000, color = 'yellow')
# ax.set_ylabel('solution time quotient')
# ax.set_xlabel('Evolution time (t/$\\tau$)')