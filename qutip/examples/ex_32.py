#
# This is a Monte-Carlo simulation showing the decay of a cavity
# Fock state |1> in a thermal environment with an average
# occupation number of n=0.063.  Here, the coupling strength is given
# by the inverse of the cavity ring-down time Tc=0.129.
#
# The parameters chosen here correspond to those from
# S. Gleyzes, et al., Nature 446, 297 (2007). 
#

#load qutip and matplotlib
from qutip import *
from pylab import *

def run():
    # define parameters
    N=4             # number of basis states to consider
    kappa=1.0/0.129 # coupling to heat bath
    nth= 0.063      # temperature with <n>=0.063

    # create operators and initial |1> state
    a=destroy(N)    # cavity destruction operator
    H=a.dag()*a     # harmonic oscillator Hamiltonian
    psi0=basis(N,1) # initial Fock state with one photon

    # collapse operators
    c_op_list = []
    # decay operator
    c_op_list.append(sqrt(kappa * (1 + nth)) * a)
    # excitation operator
    c_op_list.append(sqrt(kappa * nth) * a.dag())

    # run monte carlo simulation
    ntraj=[1,5,15,904] # list of number of trajectories to avg. over
    tlist=linspace(0,0.6,100)
    mc = mcsolve(H,psi0,tlist,c_op_list,[a.dag()*a],ntraj)
    # get expectation values from mc data (need extra index since ntraj is list)
    ex1=mc.expect[0][0]     #for ntraj=1
    ex5=mc.expect[1][0]     #for ntraj=5
    ex15=mc.expect[2][0]    #for ntraj=15
    ex904=mc.expect[3][0]   #for ntraj=904

    ## run master equation to get ensemble average expectation values ## 
    me = mesolve(H,psi0,tlist,c_op_list, [a.dag()*a])

    #  calulate final state using steadystate solver
    final_state=steadystate(H,c_op_list) # find steady-state
    fexpt=expect(a.dag()*a,final_state)  # find expectation value for particle number

    #
    # plot results using vertically stacked plots
    #
    
    # set legend fontsize
    import matplotlib.font_manager
    leg_prop = matplotlib.font_manager.FontProperties(size=10)
    
    f = figure(figsize=(6,9))
    subplots_adjust(hspace=0.001) #no space between plots
    
    # subplot 1 (top)
    ax1 = subplot(411)
    ax1.plot(tlist,ex1,'b',lw=2)
    ax1.axhline(y=fexpt,color='k',lw=1.5)
    yticks(linspace(0,2,5))
    ylim([-0.1,1.5])
    ylabel('$\left< N \\right>$',fontsize=14)
    title("Ensemble Averaging of Monte Carlo Trajectories")
    legend(('Single trajectory','steady state'),prop=leg_prop)
    
    # subplot 2
    ax2=subplot(412,sharex=ax1) #share x-axis of subplot 1
    ax2.plot(tlist,ex5,'b',lw=2)
    ax2.axhline(y=fexpt,color='k',lw=1.5)
    yticks(linspace(0,2,5))
    ylim([-0.1,1.5])
    ylabel('$\left< N \\right>$',fontsize=14)
    legend(('5 trajectories','steadystate'),prop=leg_prop)
    
    # subplot 3
    ax3=subplot(413,sharex=ax1) #share x-axis of subplot 1
    ax3.plot(tlist,ex15,'b',lw=2)
    ax3.plot(tlist,me.expect[0],'r--',lw=1.5)
    ax3.axhline(y=fexpt,color='k',lw=1.5)
    yticks(linspace(0,2,5))
    ylim([-0.1,1.5])
    ylabel('$\left< N \\right>$',fontsize=14)
    legend(('15 trajectories','master equation','steady state'),prop=leg_prop)
    
    # subplot 4 (bottom)
    ax4=subplot(414,sharex=ax1) #share x-axis of subplot 1
    ax4.plot(tlist,ex904,'b',lw=2)
    ax4.plot(tlist,me.expect[0],'r--',lw=1.5)
    ax4.axhline(y=fexpt,color='k',lw=1.5)
    yticks(linspace(0,2,5))
    ylim([-0.1,1.5])
    ylabel('$\left< N \\right>$',fontsize=14)
    legend(('904 trajectories','master equation','steady state'),prop=leg_prop)
    
    #remove x-axis tick marks from top 3 subplots
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()
    setp(xticklabels, visible=False)
    
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    xlabel('Time (sec)',fontsize=14)
    show()


if __name__=="__main__":
    run()
