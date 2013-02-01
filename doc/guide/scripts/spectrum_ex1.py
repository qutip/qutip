from qutip import *
import pylab as plt

N = 4                   # number of cavity fock states
wc = wa = 1.0 * 2 * pi  # cavity and atom frequency
g  = 0.10 * 2 * pi      # coupling strength
kappa = 0.75            # cavity dissipation rate
gamma = 0.25            # atom dissipation rate

# Jaynes-Cummings Hamiltonian
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())

# collapse operators
n_th = 0.01
c_ops = [sqrt(kappa * (1 + n_th)) * a, sqrt(kappa * n_th) * a.dag(), sqrt(gamma) * sm]

# calculate the correlation function using the mesolve
# solver, and then fft to obtain the spectrum
tlist = linspace(0, 150, 500)
corr = correlation_ss(H, tlist, c_ops, a.dag(), a)
wlist1, spec1 = spectrum_correlation_fft(tlist, corr)

# calculate the power spectrum using spectrum_ss, which
# uses essolve to solve for the dynamics
wlist2 = linspace(0.5, 1.5, 200) * 2 * pi
spec2 = spectrum_ss(H, wlist2, c_ops, a.dag(), a)

# plot the spectra
fig, ax = plt.subplots(1, 1)
ax.plot(wlist1, abs(spec1**2) / max(abs(spec1**2)),
        'b', lw=2, label='eseries method')
ax.plot(wlist2/(2*pi), abs(spec2) / max(abs(spec2)),
        'r--', lw=2, label='fft method')
ax.legend(loc=3)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power spectrum')
ax.set_title('Vacuum Rabi splitting')
ax.set_xlim(wlist2[0]/(2*pi), wlist2[-1]/(2*pi))
plt.show()
