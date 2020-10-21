######################
Fermionic Environments
######################

The basic class object used to construct the problem is imported in the following way (alongside QuTiP, with which we define system Hamiltonian and coupling operators), for the pure Python BoFiN version

.. code-block:: python

    from qutip import *
    from bofin.heom import FermionicHEOMSolver
    
If one is using the C++ BoFiN-fast package, the import is instead

.. code-block:: python

    from qutip import *
    from bofinfast.heom import FermionicHEOMSolver
    
    
Apart from this difference in import, and some additional features in the solvers in the C++ variant, the functionality that follows applies to both libraries.

One defines a particular problem instance in the following way:

.. code-block:: python

    Solver = FermionicHEOMSolver(Hsys, Q,  eta_list, gamma_list, NC,o ptions=options)


The parameters accepted by the solver are :

- ``Hsys`` : system Hamiltonian in quantum object form
- ``Q`` : a list of coupling operators (minimum two) that couple the system to the environment
- ``eta_list`` and ``gamma_list`` : the coefficients and frequencies of the correlation functions of the environment
- ``NC`` : the truncation parameter of the hierarchy
- ``options`` : standard QuTiP ``ODEoptions`` object, which is used by the ODE solver.

Note that for the Fermionic case, unlike the Bosonic case, we **don't** explicitly split the real and imaginary parts of the correlation functions. Also, for the Fermionic case, a single environment has a minimum of two coupling operators (due to the fundemental difference in how the environment interacts with the system for fermionic environments).  

A simple example of how this code works, taken from `example notebook 4b <https://github.com/tehruhn/bofin/blob/main/examples/example-4b-fermions-single-impurity-model.ipynb>`_, is:

.. code-block:: python

    %pylab inline
    from qutip import *
    from bofin.heom import FermionicHEOMSolver
    def deltafun(j,k):
        if j==k: 
            return 1.
        else:
            return 0.
    
        # Defining the system Hamiltonian      
    #Single fermion.
    d1 = destroy(2)
    #Site energy
    e1 = 1. 
    H0 = e1*d1.dag()*d1 
    #There are two leads, but we seperate the interaction into two terms, labelled with \sigma=\pm
    #such that there are 4 interaction operators (See paper)
    Qops = [d1.dag(),d1,d1.dag(),d1]

            #Bath properties:
    Gamma = 0.01  #coupling strength
    W=1. #cut-off
    T = 0.025851991 #temperature
    beta = 1./T

    theta = 2. #Bias
    mu_l = theta/2.
    mu_r = -theta/2.
            #HEOM parameters
    #Pade decompositon: construct correlation parameters

    tlist = np.linspace(0,10,200)
    lmax =10
    Alpha =np.zeros((2*lmax,2*lmax))
    for j in range(2*lmax):
        for k in range(2*lmax):
            Alpha[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)-1)*(2*(k+1)-1))
    eigvalsA=eigvalsh(Alpha)  
    eps = []
    for val in  eigvalsA[0:lmax]:
        eps.append(-2/val)
     
    AlphaP =np.zeros((2*lmax-1,2*lmax-1))
    for j in range(2*lmax-1):
        for k in range(2*lmax-1):
            AlphaP[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)+1)*(2*(k+1)+1))
                      
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

    def C(tlist,sigma,mu):
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
        c_tot = []
        for t in tlist:
            c_tot.append(sum([eta_list[l]*exp(-gamma_list[l]*t) for l in range(lmax+1)]))
        return c_tot, eta_list, gamma_list
          

    cppL,etapL,gampL = C(tlist,1.0,mu_l)
    cpmL,etamL,gammL = C(tlist,-1.0,mu_l)
    cppR,etapR,gampR = C(tlist,1.0,mu_r)
    cpmR,etamR,gammR = C(tlist,-1.0,mu_r)        
            
    Kk=lmax+1
    Ncc = 2  #For a single impurity we converge with Ncc = 2
    #Note here that the functionality differs from the bosonic case. Here we send lists of lists, were each sub-list
    #refers to one of the two coupling terms for each bath (the notation here refers to eta|sigma|L/R)

    eta_list = [etapR,etamR,etapL,etamL]
    gamma_list = [gampR,gammR,gampL,gammL]
    options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)
    resultHEOM2 = FermionicHEOMSolver(H0, Qops,  eta_list, gamma_list, Ncc,options=options)
    

Multiple environments
=====================

In dealing with multiple environments the Fermionic solver operates in a slightly different way to the Bosonic case, as already shown in the above example.
Each bath is specified by coupling to two system operators (which are related by hermitian conjugation), and the parameters for the bath coefficients associated with each of the those operators are defined in a list in the corresponding possition in ``eta_list`` and ``gamma_list``.

Typically these must be ordered in the above shown way, such that, for the first environment, ``Qops[0]`` is the operator associated with the correlation function :math:`\sigma=+`, while ``Qops[1]``  is associated with :math:`\sigma=-`.

This continues for each environment, with a corresponding set of two operators in   ``Qops``, and corresponding lists of ``etap*`` and ``etam*`` in ``eta_list``.

