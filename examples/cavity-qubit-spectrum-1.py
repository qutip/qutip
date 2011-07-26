#
# Example: Calculate the power spectrum of a cavity, with and without coupling
# to a two-level atom.
#
from qutip import *
from pylab import *

def calc_spectrum(N, wc, wa, g, kappa, gamma, tlist, wlist):

    # Hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
        
    # collapse operators
    c_op_list = []

    n_th_a = 0.5
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    A = a.dag() + a
    B = A

    # calculate the power spectrum
    corr = correlation_ss_es(H, tlist, c_op_list, A, B)

    # calculate the power spectrum
    spec = spectrum_ss(H, wlist, c_op_list, A, B)

    return corr, spec

#
# setup the calcualtion
#
N = 4              # number of cavity fock states
wc = 1.00 * 2 * pi  # cavity frequency
wa = 1.00 * 2 * pi  # atom frequency
g  = 0.10 * 2 * pi  # coupling strength
kappa = 1.0         # cavity dissipation rate
gamma = 0.2         # atom dissipation rate

wlist = linspace(0, 2*pi*2, 200)
tlist = linspace(0, 15, 500)

corr1, spec1 = calc_spectrum(N, wc, wa, g, kappa, gamma, tlist, wlist)
corr2, spec2 = calc_spectrum(N, wc, wa, 0, kappa, gamma, tlist, wlist)

figure(1)
plot(wlist/(2*pi),abs(spec1), wlist/(2*pi), abs(spec2))
xlabel('Frequency')
ylabel('Power spectrum')
legend(("g = 0.1", "g = 0.0"))

figure(2)
plot(tlist,real(corr1), tlist, real(corr2))
xlabel('Time')
ylabel('Correlation')
legend(("g = 0.1", "g = 0.0"))
show()

