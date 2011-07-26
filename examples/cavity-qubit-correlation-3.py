#
# Example: Calculate the correlation function for a cavity that is coupled to
# a qubit.
#
from qutip import *
from pylab import *

import warnings
warnings.simplefilter("error", np.ComplexWarning)


def calc_correlation(E, kappa, gamma, g, wc, w0, wl, N, tlist):
    #
    # returns the two-time correlation of the intracavity field as exponential
    # series for the problem of a coherently driven cavity with a two-level atom
    #
    # E = amplitude of driving field, kappa = mirror coupling,
    # gamma = spontaneous emission rate, g = atom-field coupling,
    # wc = cavity frequency, w0 = atomic frequency, wl = driving field frequency,
    # N = size of Hilbert space for intracavity field (zero to N-1 photons)
    #  

    # Define cavity field and atomic operators
    a  = tensor(destroy(N),qeye(2))
    sm = tensor(qeye(N),sigmam())

    # Hamiltonian
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    # collapse operators
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm.dag()

    A = a 

    corr_ode = correlation_ss_ode(H, tlist, [C1, C2], A.dag(), A)
    corr_es  = correlation_ss_es(H, tlist, [C1, C2], A.dag(), A)

    print "real corr at 0 [ode]:", corr_ode[0]
    print "real corr at 0 [es] :", corr_es[0]

    return corr_ode, corr_es

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
tlist = linspace(0,10.0,500)

start_time=time.time()
corr1, corr2 = calc_correlation(E, kappa, gamma, g, wc, w0, wl, N, tlist)
print 'time elapsed (probcorr) = ' +str(time.time()-start_time) 

figure(1)
plot(tlist,real(corr1), tlist, real(corr2))
xlabel('Time')
ylabel('Correlation <a^\dag(t)a(0)>')
legend(("ode", "es"))
show()
