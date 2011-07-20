#
# Example: Calculate the correlation function for a cavity that is coupled to
# a qubit.
#
from qutip import *
from pylab import *

def calc_correlation(E, kappa, gamma, g, wc, w0, wl, N, tlist):
    #
    # [corrES,covES] = probcorr(E,kappa,gamma,g,wc,w0,wl,N)
    #  returns the two-time correlation and covariance of the intracavity 
    #  field as exponential series for the problem of a coherently driven 
    #  cavity with a two-level atom
    #
    #  E = amplitude of driving field, kappa = mirror coupling,
    #  gamma = spontaneous emission rate, g = atom-field coupling,
    #  wc = cavity frequency, w0 = atomic frequency, wl = driving field frequency,
    #  N = size of Hilbert space for intracavity field (zero to N-1 photons)
    #  

    ida    = qeye(N)
    idatom = qeye(2)

    # Define cavity field and atomic operators
    a  = tensor(destroy(N),idatom)
    sm = tensor(ida,sigmam())

    # Hamiltonian
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    #collapse operators
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm.dag()

    return correlation_ss_ode(H, tlist, [C1, C2], a.dag(), a)
    #return correlation_ss_es(H, tlist, [C1, C2], a.dag(), a)

#
#
#
kappa = 2; 
gamma = 0.2; 
g = 5;
wc = 0; 
w0 = 0; 
wl = 0;
N = 5; 
E = 0.5;
tlist = linspace(0,10.0,200);

start_time=time.time()
corr = calc_correlation(E, kappa, gamma, g, wc, w0, wl, N, tlist);
print 'time elapsed (probcorr) = ' +str(time.time()-start_time) 

figure(1)
plot(tlist,real(corr))
xlabel('Time')
ylabel('Correlation <a^\dag(t)a(0)>')

show()
