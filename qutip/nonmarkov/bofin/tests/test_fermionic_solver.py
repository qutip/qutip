"""
Tests for the Fermionic HEOM solvers.
"""
import numpy as np
from numpy.linalg import eigvalsh
from scipy.integrate import quad
from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy, basis


    
from qutip.states import enr_state_dictionaries

from bofin.heom import FermionicHEOMSolver
from bofin.heom import _heom_state_dictionaries
import pytest
from math import sqrt, factorial

        
def test_state_dictionaries():
    """
    Tests the _heom_state_dictionaries.
    """


    #fermionic
    kcut = 6
    N_cut = 6
    nhe, he2idx, idx2he = _heom_state_dictionaries(
            [2] * kcut, N_cut
        )
        
    total_nhe = int(2**kcut
        )
    assert nhe, total_nhe
        
  
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_discrete_level_model_FermionicHEOMSolver():
    """
    FermionicHEOMSolver: compare to discrete-level current analytics
    """
    tol = 1e-3
    Gamma = 0.01  #coupling strength
    W = 1. #cut-off
    T = 0.025851991 #temperature
    beta = 1./T

    theta = 2. #Bias
    mu_l = theta/2.
    mu_r = -theta/2.
    
    tlist = np.linspace(0,10,200)


    #Pade cut-off
    lmax =10


    w_list = np.linspace(-2,2,100)
    def deltafun(j,k):
        if j==k: 
            return 1.
        else:
            return 0.
    def Gamma_L_w(w):
        return Gamma*W**2/((w-mu_l)**2 + W**2)

    def Gamma_w(w, mu):
        return Gamma*W**2/((w-mu)**2 + W**2)


    def f(x):
        kB=1.
        return 1/(np.exp(x)+1.)


    Alpha =np.zeros((2*lmax,2*lmax))
    for j in range(2*lmax):
        for k in range(2*lmax):
            Alpha[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)-1)*(2*(k+1)-1))
            
    eigvalsA=eigvalsh(Alpha)  

    eps = []
    for val in  eigvalsA[0:lmax]:
        #print(-2/val)
        eps.append(-2/val)
        
    AlphaP =np.zeros((2*lmax-1,2*lmax-1))
    for j in range(2*lmax-1):
        for k in range(2*lmax-1):
            AlphaP[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)+1)*(2*(k+1)+1))
            #AlphaP[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+2)-1)*(2*(k+2)-1))
            
    eigvalsAP=eigvalsh(AlphaP)    

    chi = []
    for val in  eigvalsAP[0:lmax-1]:
        #print(-2/val)
        chi.append(-2/val)

        
    eta_list = [0.5*lmax*(2*(lmax + 1) - 1)*( 
      np.prod([chi[k]**2 - eps[j]**2 for k in range(lmax - 1)])/
        np.prod([eps[k]**2 - eps[j]**2 +deltafun(j,k) for k in range(lmax)])) 
              for j in range(lmax)]



    kappa = [0]+eta_list

    epsilon = [0]+eps
 


    def f_approx(x):
        f = 0.5
        for l in range(1,lmax+1):
            f= f - 2*kappa[l]*x/(x**2+epsilon[l]**2)
        return f

    def C(sigma,mu):
        eta_list = []
        gamma_list  =[]
        

        eta_0 = 0.5*Gamma*W*f_approx(1.0j*beta*W)

        gamma_0 = W - sigma*1.0j*mu
        eta_list.append(eta_0)
        gamma_list.append(gamma_0)
        if lmax>0:
            for l in range(1,lmax+1):
                eta_list.append(-1.0j*(kappa[l]/beta)*Gamma*W**2/(-(epsilon[l]**2/beta**2)+W**2))
                gamma_list.append(epsilon[l]/beta - sigma*1.0j*mu)
        
        return eta_list, gamma_list

   
   
   
    


    etapL,gampL = C(1.0,mu_l)

    etamL,gammL = C(-1.0,mu_l)


    etapR,gampR = C(1.0,mu_r)
    etamR,gammR = C(-1.0,mu_r)
    
    #heom simulation with above params (Pade)
    options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

    #Single fermion.
    d1 = destroy(2)

    #Site energy
    e1 = 1. 


    H0 = e1*d1.dag()*d1 

    #There are two leads, but we seperate the interaction into two terms, labelled with \sigma=\pm
    #such that there are 4 interaction operators (See paper)
    Qops = [d1.dag(),d1,d1.dag(),d1]



    Kk=lmax+1
    Ncc = 2  #For a single impurity we converge with Ncc = 2
    
    eta_list = [etapR,etamR,etapL,etamL]
    gamma_list = [gampR,gammR,gampL,gammL]
    Qops = [d1.dag(),d1,d1.dag(),d1]

    resultHEOM2 = FermionicHEOMSolver(H0, Qops,  eta_list, gamma_list, Ncc,options=options)
    
    rhossHP2,fullssP2=resultHEOM2.steady_state()
    
    def get_aux_matrices(full, level, N_baths, Nk, N_cut, shape, dims):
        """
        Extracts the auxiliary matrices at a particular level
        from the full hierarchy ADOs.
        
        Parameters
        ----------
        full: ndarray
            A 2D array of the time evolution of the ADOs.
        
        level: int
            The level of the hierarchy to get the ADOs.
            
        N_cut: int
            The hierarchy cutoff.
        
        k: int
            The total number of exponentials used in each bath (assumed equal).
        
        N_baths: int
            The number of baths.
            
        shape : int
            the size of the ''system'' hilbert space
            
        dims : list
            the dimensions of the system hilbert space
        """
        #Note: Max N_cut is Nk*N_baths
        nstates, state2idx, idx2state = enr_state_dictionaries([2]*(Nk*N_baths) ,N_cut)#_heom_state_dictionaries([Nc + 1]*(Nk), Nc)
        aux_indices = []
        
        aux_heom_indices = []
        for stateid in state2idx:
            if np.sum(stateid) == level:
                aux_indices.append(state2idx[stateid])
                aux_heom_indices.append(stateid)
        full = np.array(full)
        aux = []

        for i in aux_indices:
            qlist = [Qobj(full[k, i, :].reshape(shape, shape).T,dims=dims) for k in range(len(full))]
            aux.append(qlist)
        return aux, aux_heom_indices, idx2state
    
    aux_1_list_list=[]
    aux1_indices_list=[]
    aux_2_list_list=[]
    aux2_indices_list=[]


    K = Kk  


    shape = H0.shape[0]
    dims = H0.dims

    aux_1_list, aux1_indices, idx2state = get_aux_matrices([fullssP2], 1, 4, K, Ncc, shape, dims)
    aux_2_list, aux2_indices, idx2state = get_aux_matrices([fullssP2], 2, 4, K, Ncc, shape, dims)


    d1 = destroy(2)   #Kk to 2*Kk
    currP = -1.0j * (((sum([(d1*aux_1_list[gg][0]).tr() for gg in range(Kk,2*Kk)]))) - ((sum([(d1.dag()*aux_1_list[gg][0]).tr() for gg in range(Kk)]))))


    
   
    
    def CurrFunc():
        def lamshift(w,mu):
            return (w-mu)*Gamma_w(w,mu)/(2*W)
        integrand = lambda w: ((2/(np.pi))*Gamma_w(w,mu_l)*Gamma_w(w,mu_r)*(f(beta*(w-mu_l))-f(beta*(w-mu_r))) /
                ((Gamma_w(w,mu_l)+Gamma_w(w,mu_r))**2 +4*(w-e1 - lamshift(w,mu_l)-lamshift(w,mu_r))**2))
        def real_func(x):
            return np.real(integrand(x))
        def imag_func(x):
            return np.imag(integrand(x))

        #These integral bounds should be checked to be wide enough if the parameters are changed
        a= -2
        b=2
        real_integral = quad(real_func, a, b)
        imag_integral = quad(imag_func, a, b)
    
   

        return real_integral[0] + 1.0j * imag_integral[0]
    
    curr_ana = CurrFunc()
    np.testing.assert_allclose(curr_ana, -currP, atol=tol)

    
        
    
