#
# E-series time evolution of an atom+cavity system.
# Adapted from a qotoolbox example by Sze M. Tan
#
from qutip import *
from pylab import *

def probevolve(E,kappa,gamma,g,wc,w0,wl,N,tlist):

    # Define cavity field and atomic operators
    a  = tensor(destroy(N),qeye(2))
    sm = tensor(qeye(N),sigmam())

    # Hamiltonian
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    #collapse operators
    C1 = sqrt(2*kappa)*a
    C2 = sqrt(gamma)*sm
    C1dC1 = C1.dag() * C1
    C2dC2 = C2.dag() * C2

    #intial state
    psi0 = tensor(basis(N,0),basis(2,1))
    rho0 = ket2dm(psi0)

    # Calculate the Liouvillian
    L = liouvillian(H, [C1, C2])

    # Calculate solution as an exponential series
    start_time = time.time()
    rhoES = ode2es(L,rho0);
    print 'time elapsed (ode2es) = ' + str(time.time()-start_time) 
    
    # Calculate expectation values
    start_time = time.time()
    count1  = esval(expect(C1dC1,rhoES),tlist);
    count2  = esval(expect(C2dC2,rhoES),tlist);
    infield = esval(expect(a,rhoES),tlist);
    print 'time elapsed (esval) = ' +str(time.time()-start_time) 

    # alternative
    start_time = time.time()
    expt_list = essolve(H, psi0, tlist, [C1, C2], [C1dC1, C2dC2, a]) 
    print 'time elapsed (essolve) = ' +str(time.time()-start_time) 

    return count1, count2, infield, expt_list[0], expt_list[1], expt_list[2]


#-------------------------------------------------------------------------------
# setup the calculation
#-------------------------------------------------------------------------------
kappa = 2; 
gamma = 0.2;
g  = 1; 
wc = 0; 
w0 = 0; 
wl = 0;
N  = 4; 
E  = 0.5;

tlist = linspace(0,10,200);

start_time = time.time()
count1, count2, infield, count1_2, count2_2, infield_2 = probevolve(E,kappa,gamma,g,wc,w0,wl,N,tlist);
print 'time elapsed = ' + str(time.time()-start_time) 

plot(tlist, real(count1), tlist, real(count1_2), '.')
plot(tlist, real(count2), tlist, real(count2_2), '.')
xlabel('Time')
ylabel('Transmitted Intensity and Spontaneous Emission')
show()

