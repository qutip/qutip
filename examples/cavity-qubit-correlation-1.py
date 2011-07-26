
from qutip import *
from pylab import *

#
#
#
def probcorr(E,kappa,gamma,g,wc,w0,wl,N):
    #
    # returns the two-time correlation and covariance of the intracavity 
    # field as exponential series for the problem of a coherently driven 
    # cavity with a two-level atom
    #
    # E = amplitude of driving field, kappa = mirror coupling,
    # gamma = spontaneous emission rate, g = atom-field coupling,
    # wc = cavity frequency, w0 = atomic frequency, wl = driving field frequency,
    # N = size of Hilbert space for intracavity field (zero to N-1 photons)
    #  

    # Hamiltonian
    a  = tensor(destroy(N),qeye(2))
    sm = tensor(qeye(N),destroy(2))
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    # collapse operators
    C1 = sqrt(2*kappa) * a
    C2 = sqrt(gamma)   * sm.dag()
    c_op_list = [C1, C2]

    # Calculate the Liouvillian
    L = liouvillian(H, c_op_list)

    # Find steady state density matrix and field
    rhoss = steady(L);

    ass = expect(a,rhoss);
 
    # Initial condition for regression theorem
    arho = a*rhoss;

    # Solve differential equation with this initial condition
    solES = ode2es(L,arho);
   
    # Find trace(a' * solution)
    corrES = expect(a.dag(),solES);

    # Calculate the covariance by subtracting product of means
    covES = corrES - real(conjugate(ass)*ass)

    return corrES, covES


#
#
#
kappa = 2
gamma = 0.2
g = 5
wc = 0 
w0 = 0 
wl = 0
N = 5 
E = 0.5

start_time=time.time()
corrES,covES = probcorr(E,kappa,gamma,g,wc,w0,wl,N);
print 'time elapsed (probcorr) = ' +str(time.time()-start_time) 

tlist = linspace(0,10.0,500);

start_time=time.time()
corr  = esval(corrES,tlist);
cov   = esval(covES,tlist);
print 'time elapsed (esval) = ' +str(time.time()-start_time) 



figure(1)
plot(tlist,real(corr))
plot(tlist,real(cov))
xlabel('Time')
ylabel('Correlation and covariance')
legend(("corr", "cov"))
wlist = linspace(-10,10,200);

start_time=time.time()
Ps = esspec(covES,wlist);
print 'time elapsed (esspec) = ' +str(time.time()-start_time) 

figure(2)
plot(wlist,real(Ps))
xlabel('Frequency')
ylabel('Spectral amplitude')
show()


