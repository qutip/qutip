# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:35:37 2023

@author: Fenton
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:42:12 2023

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

power_range = 1
P_array = np.zeros(power_range)
for i in range(power_range):
    P_array[i]=((0+i))

Z1 = []
Z2 = []
Z3 = []


for idz, E2P in enumerate(P_array):
    print('Working on Spectra number', idz, 'of',len(P_array))
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
    d = dmag *  np.sqrt(1/2) * np.array([1, -1j, 0]) 
    
    Om1  = np.dot(d,        E1)
    Om1t = np.dot(d,np.conj(E1))
    
   
    wlas = wres-(Om1*(3-idz))
   
    T = 2*np.pi/abs(wres) # period of the Hamiltonian
    Hargs = {'l': (wres)}                           #Characteristic frequency of the Hamiltonian is half the beating frequency of the Hamiltonian after the RWA. QuTiP needs it in Dictionary form.
    w = Hargs['l']
    
    '''
    Defining the spontaneous emission operators that will link the two states
    Another paper I've looked up seems to say that decay values of ~275 kHz
        are common. I think.
    '''
    Gamma = 2 * np.pi * wres*.03   #in THz, roughly equivalent to 1 micro eV
    spont_emis = np.sqrt(Gamma) * mat(0,1)           # Spontaneous emission operator 
    
    #symmertric spectral density
    spont_emis1 = Gamma          
    

      
        
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
    Ntau =  (Nt)*2000                                 #50 times the number of points of tlist
    taume = (Ntau/Nt)*T                               #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau = taume/Ntau                                 #time spacing in taulist - same as tlist!
    taulist = np.linspace(0, taume-dtau, Ntau)        #Combining everything to make taulist, and I want taulist to end exactly at the beginning/end of a period to make some math easier on my end later
    
    Ntau2 = (Nt)*100                                #50 times the number of points of tlist
    taume2 = (Ntau2/Nt)*T                             #taulist goes over 50 periods of the system so that 50 periods can be simulated
    dtau2 = taume2/Ntau2                              #time spacing in taulist - same as tlist!
    taulist2 = np.linspace(0, taume2-dtau2, Ntau2)   
    
    timey_wimey = 0
    '''
    Creating the matrix of omega values that will be used in the fft
    '''
    omega_array1 = np.fft.fftfreq(Ntau2,dtau2)
    omega_array = np.fft.fftshift(omega_array1)


    '''
    Finally, I'll define the Hamiltonian
    I'll have a diagonal time independant part and two offdiagonal terms. 
    
    Both offdiagonal terms will have forward and backward rotating parts even after
        the RWA due to the interaction of the two lasers with the dot
    '''
    
    
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
    
    
    

    FlimeSolve1 = flimesolve(
        Htot,
        rho0,
        tlist,
        taulist,
        c_ops=[destroy(2)],
        c_op_rates=[spont_emis1],
        T=T,
        args=Hargs,
        time_sense=timey_wimey,
        quicksolve = False,
        options={"normalize_output": False})
    rhoss1 = TimeEvol1.states[-1] #Calling the longest time evolved output state from the first mesolve
    QuickSolve1 = flimesolve(
        Htot,
        rho0,
        tlist,
        np.array([taulist[-1]]),
        c_ops=[destroy(2)],
        c_op_rates=[spont_emis1],
        T=T,
        args=Hargs,
        time_sense=timey_wimey,
        quicksolve = True,
        options={"normalize_output": False})
    rhoss2 = TimeEvol2.states[-1] #Calling the longest time evolved output state from the first mesolve
    TimeEvol2 = mesolve(
        Htot,
        rho0,
        taulist,
        c_ops=[np.sqrt(Gamma)*destroy(2)],
        options={"normalize_output": False},
        args=Hargs,
        )
    
    flimestates = np.stack([i.full() for i in TimeEvol1.states])
    # mestates = np.stack([i.full() for i in TimeEvol2.states])
    
    fig, ax = plt.subplots(2,1)                                                    

    ax[0].plot(taulist/T,np.sqrt((flimestates[:,0,0])**2), color = 'blue' )
    ax[0].plot(taulist/T,np.sqrt((mestates[:,0,0])**2), color = 'black',linestyle = '--' )
    ax[0].legend(['FLiMESolve','MEsolve'])
    ax[0].set_ylabel('Ground State Population')  

    # ax[1].plot(taulist/T,np.sqrt((flimestates[:,1,1])**2), color = 'blue' )
    # ax[1].plot(taulist/T,np.sqrt((mestates[:,1,1])**2), color = 'black',linestyle = '--' )
    # ax[1].set_xlabel('time (t/T)') 
    # ax[1].set_ylabel('Excited State Population')







#     '''
#     Next step is to iterate this steady state rho_s forward in time. I'll choose the times
#     to be evenly spread out within T, the time scale of the Hamiltonian
    
#     Also going through one time periods of the Hamiltonian so that I can graph the states
#     and make sure I'm in the limit cycle
#     '''
#     PeriodStates1 = flimesolve(
#         Htot,
#         rhoss1,
#         taulist[-1]+dtau+tlist,
#         c_ops=[destroy(2)],
#         c_op_rates=[spont_emis1],
#         T=T,
#         args=Hargs,
#         time_sense=timey_wimey,
#         options={"normalize_output": False})
#     PeriodStates2 = flimesolve(
#         Htot,
#         rhoss1,
#         taulist[-1]+dtau+tlist,
#         c_ops=[destroy(2)],
#         c_op_rates=[spont_emis1],
#         T=T,
#         args=Hargs,
#         time_sense=timey_wimey,
#         quicksolve = True,
#         options={"normalize_output": False})

#     # PeriodStates3 = flimesolve(
#     #     Htot3,
#     #     rhoss3,
#     #     taulist[-1]+dtau+tlist,
#     #     c_ops=[destroy(2)],
#     #     c_op_rates=[spont_emis3],
#     #     T=T,
#     #     args=Hargs,
#     #     time_sense=timey_wimey,
#     #     options={"normalize_output": False})
   


    
#     '''
#     Next I'll multiply by the lowering operator
#     '''
#     Bstates1 = [ Qobj(None, [[Hdim],[Hdim]]) ] * len(tlist)
#     Bstates2 = [ Qobj(None, [[Hdim],[Hdim]]) ] * len(tlist)
#     # Bstates3 = [ Qobj(None, [[Hdim],[Hdim]]) ] * len(tlist)
    
#     for idx in range(len(tlist)):
#         Bstates1[idx] = Qobj(PeriodStates1.states[idx]*destroy(2).dag())
#         Bstates2[idx] = Qobj(PeriodStates2.states[idx]*destroy(2).dag())
#         # Bstates3[idx] = Qobj(mat(0,1)*PeriodStates3.states[idx])
       

#     '''
#     Setting up a matrix to have rows equal to the number of tau values and columns equal to the number of t values
#     At the end I'll average over each row to get a vector where each entry is a tau value and an averaged t value
#     '''
    
#     Astates1 = np.zeros( (len(taulist2), len(Bstates1),2,2), dtype='complex_' ) 
#     Astates2 = np.zeros( (len(taulist2), len(Bstates2),2,2), dtype='complex_' ) 
#     # Astates3 = np.zeros( (len(taulist2), len(Bstates3),2,2), dtype='complex_' ) 
#     for tdx in range(len(tlist)): #First for loop to find the tau outputs for each t value
#         print('Filling column',tdx+1,'of',len(Bstates1))
            
#         taulist2 = np.round(taulist[-1]+dtau+tlist[idx]+taulist2,8)
#         TauBSEvol1 = flimesolve(
#             Htot,
#             Bstates1[tdx],
#             taulist2,
#             c_ops=[mat(0,1)],
#             c_op_rates=[spont_emis1],
#             T=T,
#             args=Hargs,
#             time_sense=timey_wimey,
#             options={"normalize_output": False})
#         TauBSEvol2 = flimesolve(
#             Htot,
#             Bstates1[tdx],
#             taulist2,
#             c_ops=[mat(0,1)],
#             c_op_rates=[spont_emis1],
#             T=T,
#             args=Hargs,
#             time_sense=timey_wimey,
#             quicksolve = True,
#             options={"normalize_output": False})
#         # TauBSEvol3 = flimesolve(
#         #     Htot3,
#         #     Bstates3[tdx],
#         #     taulist2,
#         #     c_ops=[mat(0,1)],
#         #     c_op_rates=[spont_emis3],
#         #     T=T,
#         #     args=Hargs,
#         #     time_sense=timey_wimey,
#         #     options={"normalize_output": False})
       
#         for taudx in range(len(taulist2)): #Second loop to set the elements of the matrix
#                 Astates1[taudx,tdx]=(destroy(2)*TauBSEvol1.states[taudx]).full()
#                 Astates2[taudx,tdx]=(destroy(2)*TauBSEvol2.states[taudx]).full()
#                 # Astates3[taudx,tdx]=(mat(1,0)*TauBSEvol3.states[taudx]).full()
              
    
#     '''
#     Okay so the output matrix from above is a bunch of 2x2 density matrices
#     where the value idx1 refers to the tau value and the value idx refers to the t value
    
#     Going forward I should now average over each "row" of t values, i.e. average over idx
#     '''

#     Avg1 = np.mean(Astates1,axis=1)
#     Avg2 = np.mean(Astates2,axis=1)
#     # Avg3 = np.mean(Astates3,axis=1)
   
    
    
#     g11 = np.zeros((len(Avg1)), dtype='complex_')
#     g12 = np.zeros((len(Avg2)), dtype='complex_')
#     # g13 = np.zeros((len(Avg3)), dtype='complex_')
   
#     for taudx in range(len(Avg1)):
#         g11[taudx] = np.trace(Avg1[taudx])
#         g12[taudx] = np.trace(Avg2[taudx])
#         # g13[taudx] = np.trace(Avg3[taudx])
        
#     spec1 = np.fft.fft(g11,axis=0)
#     spec1 = np.fft.fftshift(spec1)
    
#     spec2 = np.fft.fft(g12,axis=0)
#     spec2 = np.fft.fftshift(spec2)
    
#     # spec3 = np.fft.fft(g13,axis=0)
#     # spec3 = np.fft.fftshift(spec3)
    
    
#     Flimespec = np.real(spec1)/len(g11)
#     MESpec = np.real(spec2)/len(g12)
#     # Z3[idz,:] = np.real(spec3)/len(g13)                          

#     Z1.append(Flimespec)
#     Z2.append(MESpec)

# # Plot on a colorplot
# fig, ax = plt.subplots(1,1)
# limits = [omega_array[0]-(w/(2*np.pi)),\
#           omega_array[-1]-(w/(2*np.pi)),\
#           P_array[0],\
#           P_array[-1]]
# pos = ax.imshow(Z1,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = [1e-6,1e-2]) 

# fig.colorbar(pos)
# ax.set_xlabel('$\omega_{res}-\omega$ [THz]')
# ax.set_ylabel("$\u03A9_{2} (\u03BCeV)$") 

# # Plot on a colorplot
# fig, ax = plt.subplots(1,1)
# limits = [omega_array[0]-(w/(2*np.pi)),\
#           omega_array[-1]-(w/(2*np.pi)),\
#           P_array[0],\
#           P_array[-1]]
# pos = ax.imshow(Z2,cmap=plt.get_cmap(cm.bwr), aspect='auto', interpolation='nearest', origin='lower',
#             extent = limits,  norm=matplotlib.colors.LogNorm(), clim = [1e-6,1e-2]) 

# fig.colorbar(pos)
# ax.set_xlabel('$\omega_{res}-\omega$ [THz]')
# ax.set_ylabel("$\u03A9_{2} (\u03BCeV)$") 

     
# # freqlims = [omega_array[0],omega_array[-1]]

# # frequency_range = (omega_array-(w/(2*np.pi)))
# # idx0 = np.where(abs(frequency_range-freqlims[0]) == np.amin(abs((frequency_range-freqlims[0] ))))[0][0]
# # idxf = np.where(abs(frequency_range-freqlims[1]) == np.amin(abs((frequency_range-freqlims[1] ))))[0][0]

# # plot_freq_range = frequency_range[idx0:idxf]
# # Z1_truncd =Z1[0][idx0:idxf]
# # Z2_truncd =Z2[0][idx0:idxf]
# # # Z3_truncd =Z3[0][idx0:idxf]

# # fig, ax = plt.subplots(2,1)                                                    #Plotting the results!
# # # ax.semilogy( omega_array-(w/(2*np.pi)), Z1[0], color = 'r' )
# # ax[0].semilogy( omega_array+(w/(2*np.pi)), Z1[0], color = 'black')
# # ax[0].legend(['Floquet'])
# # ax[1].semilogy( omega_array+(w/(2*np.pi)), Z2[0], color = 'red')
# # ax[1].legend(['Direct Integration'])
# # # ax.plot( (plot_freq_range), Z3_truncd, color = 'green')

# # ax[1].set_xlabel("$\omega_{res}$-$\Delta$ (THz)")
# # # ax.axvline(x=(-2), color='k', linestyle = 'dashed')
# # # ax.axvline(x=(2), color='k', linestyle = 'dashed')

# # # ax.set_xlabel(F"3LS with $\Omega_1$ polarization = {P2} THz {L2pol} polarization and swept detuning (detuning is from upper resonance)")
# # ax[0].set_ylabel("Amplitude") 
# # ax[1].set_ylabel("Amplitude") 

