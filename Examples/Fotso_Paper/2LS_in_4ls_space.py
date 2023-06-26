# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 23:41:26 2023

@author: Fenton
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:13:36 2023

@author: Fenton
"""
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:17:41 2023

@author: Fenton
"""


import matplotlib.pyplot as plt
from qutip import flimesolve,mesolve,Qobj,basis,destroy,correlation
import numpy as np
# from qutip import *
# from qutip.ui.progressbar import BaseProgressBar
import matplotlib.cm as cm #importing colormaps to plot nicer
import matplotlib


from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')




############## Experimentally adjustable parameters #####################
#electric field definitions


E1mag = 2*np.pi*200
'''
Defining the polarization that will dot with the dipole moment to form the
Rabi Frequencies
'''
E1pol = 1
#np.sqrt(1/2)*np.array([1, 1, 0]); 
   

'''
Total E field
'''
E1 = E1mag*E1pol
   
############## Hamiltonian parameters ###################################


'''
Defining the spontaneous emission operators that will link the two states
Another paper I've looked up seems to say that decay values of ~275 kHz
    are common. I think.
'''
Gamma = E1mag*.03#2*np.pi*2.5e-3   #in THz, roughly equivalent to 1 micro eV


'''
Defining the Dipole moment of the QD that will dot with the laser polarization
    to form the Rabi Frequency and Rabi Frequency Tilde
'''
dmag = 1
#Leaving below in as a comment in case I want to implement polarization later
# d = dmag *  np.sqrt(1/2) * np.array([1, 1, 0]) 
d1 = 1
d2 = 1

Om_dot_1  = np.dot(d1,        E1)
Om_dot_1t = np.dot(d1,np.conj(E1))
Om_dot_2  = np.dot(d2,        E1)
Om_dot_2t = np.dot(d2,np.conj(E1))  

wlas = 2*np.pi*280 #THz
wres1 = wlas + 0*Gamma #Detuned 3Gamma blue
wres2 = wlas - 40*Gamma #Detuned 3Gamma red
   
Delta_1 = wres1-wlas
Delta_2 = wres2-wlas
   
T = 2*np.pi/abs(wlas) # period of the Hamiltonian
Hargs = {'l': (wlas)}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.
w = Hargs['l']


  
    
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

Ntau =  int((Nt)*1e+3)                                 #50 times the number of points of tlist
taume = (Ntau/Nt)*T                               #taulist goes over 50 periods of the system so that 50 periods can be simulated
dtau = taume/Ntau                                 #time spacing in taulist - same as tlist!
taulist = np.linspace(0, taume-dtau, Ntau)        #Combining everything to make taulist, and I want taulist to end exactly at the beginning/end of a period to make some math easier on my end later
   
 
Ntau2 = (Nt)*10000                               #50 times the number of points of tlist
taume2 = (Ntau2/Nt)*T                             #taulist goes over 50 periods of the system so that 50 periods can be simulated
dtau2 = taume2/Ntau2                              #time spacing in taulist - same as tlist!
taulist2 = np.linspace(0, taume2-dtau2, Ntau2)   

omega_array1 = np.fft.fftfreq(Ntau2,dtau)
omega_array = np.fft.fftshift(omega_array1)

lower_dot_1 = Qobj(np.array([[0,1,0,0],
                             [0,0,0,0],
                             [0,0,0,0],
                             [0,0,0,0]
                             ]))
# lower_dot_2 = Qobj(np.array([[0,0,0,0],
#                              [0,0,0,0],
#                              [0,0,0,1],
#                              [0,0,0,0]
#                              ]))

          
################################# Hamiltonian #################################
'''
Finally, I'll define the full system Hamiltonian. Due to the manner in which
    QuTiP accepts Hamiltonians, the Hamiltonian must be defined in separate terms
    based on time dependence. Due to that, the Hamiltonian will have three parts.
    One time independent part, one part that rotates forwards in time, and one
    part that rotates backwards in time. For full derivation, look at my "2LS
    Bichromatic Excitation" subtab under my 4/13/21 report
'''
   

H_atom = (1/2)*np.array([[-wres1,      0,       0,      0],
                         [       0,wres1,       0,      0],
                         [       0,      0,     0,      0],
                         [       0,      0,      0,     0]
                         ])

Hf1  = np.array([[                 0,Om_dot_1,                  0,        0],
                 [np.conj(Om_dot_1t),       0,                  0,        0],
                 [                 0,       0,                  0,        0],
                 [                 0,       0,                  0,        0]
                 ])

Hb1 =  np.array([[                0,Om_dot_1t,                  0,         0],
                 [np.conj(Om_dot_1),        0,                  0,         0],
                 [                0,        0,                  0,         0],
                 [                0,        0,                  0,         0]
                 ])

  

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

rho0 = basis(4,0)+basis(4,2)

'''
Next step is to iterate it FAR forward in time, i.e. far greater than any timescale present in the Hamiltonian

Longest timescale in the Hamiltonian is Delta=.02277 Thz -> 5E-11 seconds

Going to just use 1 second as this should be more than long enough. Might need 
to multiply by the conversion factor as everything so far is in Thz...

jk just gunna multiply the time scale by a big number

It requires a tlist for some reason so I'll just take the last entry for the next stuff
'''


TimeEvolF = flimesolve(
        Htot,
        rho0,
        taulist,
        c_ops_and_rates = [[lower_dot_1,Gamma]],
        T = T,
        args = Hargs,
        time_sense = 1,
        quicksolve = False,
        options={"normalize_output": False})
rhossF = TimeEvolF.states[-1]

'''
Next step is to iterate this steady state rho_s forward in time. I'll choose the times
to be evenly spread out within T, the time scale of the Hamiltonian

Also going through one time periods of the Hamiltonian so that I can graph the states
and make sure I'm in the limit cycle
'''


PeriodStatesF = flimesolve(
        Htot,
        rhossF,
        taulist[-1]+tlist,
        c_ops_and_rates = [[lower_dot_1,Gamma]],
        T = T,
        args = Hargs,
        time_sense = 1,
        quicksolve = False,
        options={"normalize_output": False})


testg1 = np.zeros((len(tlist), len(taulist2)), dtype='complex_' ) 
for tdx in range(len(tlist)):
    '''
    Start here tomorrow. You need to write taulist into the _make_solver
    arguments in Correlation, so that the FLiMESolver can construct
    properly. Then, since I'm probably dropping the automatic timer averaging,
    I'll need to use the for loop (for tdx in range(len(tlist)):) to calculate
    all the different g1s and then average them.'
    '''

    testg1[tdx] = correlation.correlation_2op_1t(Htot,
                                                  PeriodStatesF.states[tdx],
                                                  taulist = taulist[-1]+tlist[tdx]+taulist2,
                                                  c_ops=[[lower_dot_1,Gamma]],
                                                  a_op = (lower_dot_1).dag(),
                                                  b_op = (lower_dot_1),
                                                  solver="fme",
                                                  reverse = True,
                                                  options = {'T':T},
                                                  args = Hargs)[0]

g1avg = np.average(testg1,axis=0)
specF = np.fft.fft(g1avg,axis=0)
specF = np.fft.fftshift(specF)/len(g1avg)

ZF = specF


fig, ax = plt.subplots(1,1)                                                    #Plotting the results!
ax.semilogy( (omega_array+(wlas/(2*np.pi))), ZF, color = 'r' )
ax.set_xlabel('Detuning [THz]')
ax.set_ylabel("Amplitude") 
ax.set_title(r'Heavily Driven 2LS at resonance with $E_{mag} = 0.5 \omega_{res}$' )
ax.legend(['Mollow Triplet From Correlation Function'])

