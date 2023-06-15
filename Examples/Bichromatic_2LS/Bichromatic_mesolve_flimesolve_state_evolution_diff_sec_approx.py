# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:36:08 2023

@author: Fenton
"""


import matplotlib.pyplot as plt
import numpy as np
import time # Lets us time execution of code sections
import scipy as scp
from qutip import basis,flimesolve,mesolve,destroy,Qobj, correlation
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib.pyplot as plt
import matplotlib.colors
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

start_time = time.time()

def mat(i,j):
    return(basis(2,i)*basis(2,j).dag())
'''
power range is how many spectra you want to do. 

Change the constant in period_array to change the starting offresonant laser
    Rabi frequency (in mu-eV)
''' 


period_array = [50]#np.linspace(1,50,100)
fsolve_time_array0 = []
fsolve_time_array1 = []
fsolve_time_array10 = []
fsolve_time_array100 = []
fsolve_time_array1000 = []
msolve_time_array = []
for idz, periods in enumerate(period_array):
    print('Working on Spectra number', idz+1, 'of', len(period_array))
    ############## Experimentally adjustable parameters ##########################
    '''
    The purpose of this section is to set up the laser and dipole moment parameters
        for use in the Hamiltonian
    
    The magnitude and polarization of the lasers and dipole moment must be defined. 
        Additionally, the laser frequencies must be defined.These paramaters control 
        the steady state population of the system, among other things (I think).
        
    I use "E" for the lasers because I'm defining their electric fields'
    
    As of 5/28/22, I've gone back and switched L1 and L2 such that the characteristic
        frequency, w=beat/2, becomes positive. This doesn't change the correct results
        but simplifies some issues with signs that I wasn't sure about later on. Don't
        think it will be necessary to switch back ever, but having the robust option
        to do so might be nice...
    '''
    
    E2mag = 2*np.pi*0.0072992700729927                #Don't remember why we use this number but it's important. Related to experimental values! Maybe Gustin's?
    E1mag = 2*np.pi*60*0.0072992700729927/30                               
    dmag  = 1                                         #Dipole moment magnitude. Setting to 1 because it makes things simpler
    
    '''
    Defining the polarizations in the form of [x,y,z]. The polarizations of the lasers 
        and dipole moment are used to find the Rabi Frequencies below.
    '''
    
    E2pol = np.sqrt(1/2)*np.array([1, 1, 0])                       #Laser 1 Polarization
    E1pol = np.sqrt(1/2)*np.array([1, 1, 0])                       #Laser 2 Polarization
    dpol  = np.sqrt(1/2)*np.array([1, 1, 0])                       #Dipole Moment Polarization
    
    '''
    Next defining the two laser angular frequencies
    
    w2 detuning is chosen because at the time this is written, the eventual goal of
        this code is to reproduce Gustin's "High-resolution spectroscopy of a quantum 
        dot driven bichromatically by two strong coherent fields"
    '''
    w2 = 2*np.pi*280                            #Laser 1 Frequency (THz)
    w1 = 2*np.pi*279.99275                                         #Laser 2 Frequency (THz). 0.00725 THz Detuning is ~30 mueV, same as Gustin's detuning
    
    
    '''
    Total Electric field from the lasers are the product of the laser magnitudes and 
        polarization. The dipole moment (of the simulated quantum dot) is defined as 
        the product of its magnitude and polarization. They're vectors with a direction.
    '''
    E1 = E1mag*E1pol                                  #Laser 1 Electric Field
    E2 = E2mag*E2pol                                  #Laser 2 Electric Field
    d  = dpol *dmag                                   #Dipole Moment of the QD
    
    
    ############## Hamiltonian parameters ########################################
    '''
    This section contains parameters and derived numbers that are necessary for the
        definition of the system Hamiltonian. These are the Rabi Frequencies of 
        the laser/dot interaction, the definition of the Hilbert space and energy 
        levels used to simulate the Quantum Dot, the lowering operator (and its
        stregth) that represents spontaneous emission, and some mathematical 
        constants that make defining the Hamiltonian a bit less messy. Finally, I 
        define two time lists that are necessary for the simulation.
    '''
    
    '''
    The rabi frequencies are defined as the dot product of each laser's electric
        field and the dipole moment. These are used in the Hamiltonian below
    '''
    Om1  = np.dot(d,E1)                               #Rabi Frequency Omega_1
    Om2  = np.dot(d,E2)                               #Rabi Frequency Omega_2
    
    
    '''
    Defining the dimension of the system and the energy levels
    '''
    Hdim = 2                                          #dimension of Hilbert space - 2 energy levels
    
    gnd0 = basis(2,0)                                 #|0> - Ground State
    wgnd = 2*np.pi*0                                  #Ground state angular frequency
    
    gnd1 = basis(2,1)                                 #|1> - Excited State
    wres = 2*np.pi*280                                #Excited state frequency
    
    
    '''
    Defining useful constants. These make the Hamiltonian definition in the next
        section look a whole lot nicer, and are used often enough in the rest of the
        script that it's nice to have them defined in one place. 
    '''
    Delta =wres-(1/2)*(w1+w2)                        #Average Detuning of the lasers from the excited state
    beat  = abs(w2-w1)                                   #beat frequency, difference between offresonant laser (L2) and resonant laser (L1)
    T     = (2*np.pi/abs(beat/2))                     #Characteristic Time period of the Hamiltonian - defined with abs(beat/2) as beat/2 is negative. 
                                                      #5/27/22 - MIGHT NEED TO SOLVE T without abs(beat/2). DON'T KNOW HOW THAT WILL AFFECT THINGS
    Hargs = {'l': (beat/2)}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.
    w = Hargs['l']                                    #Numerical value of the characteristic frequency. Useful for later calculations
    
                                       
    '''
    Defining the spontaneous emission operators that will link the ground and excited 
        states. Magnitude from Gustin
    '''
    Gamma  = 2 * np.pi * .00025                       #Magnitude of the lowering operator, in 1/fs or THz.
    low_op =  mat(0,1)                                #lowering operator that acts on the excited state to bring it to the ground state
 
    
    '''
    Defining the two time lists necessary for the experiment
    
    t list is used to solve for the Floquet modes, which only require one period
        of the system to be fully solved, and also which are used at various points
        in the rest of this code
    
    taulist is used to actually time evolve the system forward, and also to solve
        for the Floquet states.
    
    I define them to have the same exact time spacing because it makes the solutions
        look nicer. Or maybe I just think it does. IIRC I defined the state table
        with some simple interpolation so maybe a mismatch wouldn't be that bad.
        
    Using a coarse step size for now to make it solve much faster. Will improve
        resolution later!
    '''
    Nt = (2**8)                                       #Number of Points
    ttime = T                                          #Length of time of tlist defined to be one period of the system
    dt = ttime/Nt                                      #Time point spacing in tlist
    tlist = np.linspace(0, ttime-dt, Nt)               #Combining everything to make tlist
     
                                                      #Taulist Definition
    Ntau =  int((Nt)*periods)                                   #50 times the number of points of tlist
    taume = (Ntau/Nt)*T                               #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau = taume/Ntau                                 #time spacing in taulist - same as tlist!
    taulist = np.linspace(0, taume-dtau, Ntau)        #Combining everything to make taulist, and I want taulist to end exactly at the beginning/end of a period to make some math easier on my end later
    
    Ntau2 = (Nt)*500                                #50 times the number of points of tlist
    taume2 = (Ntau2/Nt)*T                             #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau2 = taume2/Ntau2                              #time spacing in taulist - same as tlist!
    taulist2 = np.linspace(0, taume2-dtau2, Ntau2)   

    
    if idz == 0:
        print('making Z array')
        omega_array1 = np.fft.fftfreq(Ntau2,dtau)
        omega_array = np.fft.fftshift(omega_array1)
        
        ZF = np.zeros( (len(period_array), len(omega_array1)) )
       

        ################################# Hamiltonian #################################
    '''
    Finally, I'll define the full system Hamiltonian. Due to the manner in which
        QuTiP accepts Hamiltonians, the Hamiltonian must be defined in separate terms
        based on time dependence. Due to that, the Hamiltonian will have three parts.
        One time independent part, one part that rotates forwards in time, and one
        part that rotates backwards in time. For full derivation, look at my "2LS
        Bichromatic Excitation" subtab under my 4/13/21 report
    '''
    
    
    H0 =  (1/2)*Delta*(-mat(0,0)+mat(1,1))                                #Time independant Term
    H1 = -(1/2)*(Om2*mat(0,1)+np.conj(Om1)*mat(1,0))                      #Forward Rotating Term
    H2 = -(1/2)*(Om1*mat(0,1)+np.conj(Om2)*mat(1,0))                      #Backward Rotating Term
    
    H = [H0,                                        \
        [H1,'exp(1j * l * t )'],                    \
        [H2, 'exp(-1j * l * t )']]                                        #Full Hamiltonian in string format, a form acceptable to QuTiP
    
    
    
    print('finished setting up the Hamiltonian')

    ############### Time evolving rho0 with solve_ivp#########################
    '''
    The purpose of this section is to actually time evolve rho0 and then plot the 
        results. The ODE solver takes the rhodot function above, the rho0 in the
        FLOQUET STATE basis, and then time evolved for a time length taulist. Next, it
        takes the result in the FLOQUET STATE basis, turns it into a superoperator,
        and moves it back to the COMPUTATIONAL basis. Finally, it plots the results
        as expectation values of the various density operator matrix element expectation
        values.
    '''
    
    
    '''
         This loop takes each array output by the solver,
        turns it into a quantum object with the appropriate shape and dimensions to be 
        considered a "operator ket" type operator in QuTiP, then uses QuTiP to do a basis
        transformation back to the COMPUTATIONAL BASIS
    '''
    print('attempting to solve the IVP')    
    rho0 = basis(2,0)
   
    
    TimeEvolF0 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 0,
            quicksolve = False,
            options={"normalize_output": False})
    fsolve_time_array0.append(TimeEvolF0.stats["run time"])
    
    TimeEvolF1 = flimesolve(
            H,
            rho0,
            taulist,
            c_ops_and_rates = [[destroy(2),Gamma]],
            T = T,
            args = Hargs,
            time_sense = 1,
            quicksolve = False,
            options={"normalize_output": False})
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
            options={"normalize_output": False})
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
            options={"normalize_output": False})
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
            options={"normalize_output": False})
    fsolve_time_array1000.append(TimeEvolF1000.stats["run time"])

    TimeEvolM =  mesolve(
            H,
            rho0,
            taulist,
            c_ops=[np.sqrt(Gamma)*destroy(2)],
            options={"normalize_output": False,
                     'atol':1e-16,
                     'rtol':1e-16},
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
# ax.legend(['Floquet excited State Population'])
# ax[1].legend(['Direct Integration Excited State Population'])
ax.set_xlabel('Evolution time (t/$\\tau$)')

perc_dev0 = np.sqrt(np.average(((mstates[1::,1,1]-fstates0[1::,1,1])/mstates[1::,1,1])**2))
perc_dev1 = np.sqrt(np.average(((mstates[1::,1,1]-fstates1[1::,1,1])/mstates[1::,1,1])**2))
perc_dev10 = np.sqrt(np.average(((mstates[1::,1,1]-fstates10[1::,1,1])/mstates[1::,1,1])**2))
perc_dev100 = np.sqrt(np.average(((mstates[1::,1,1]-fstates100[1::,1,1])/mstates[1::,1,1])**2))
perc_dev1000 = np.sqrt(np.average(((mstates[1::,1,1]-fstates1000[1::,1,1])/mstates[1::,1,1])**2))
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