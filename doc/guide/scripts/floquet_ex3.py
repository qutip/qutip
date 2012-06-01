from qutip import *

delta = 0.0  * 2*pi; eps0  = 1.0 * 2*pi
A     = 0.25 * 2*pi; omega = 1.0 * 2*pi
T      = (2*pi)/omega 
tlist  = linspace(0.0, 10 * T, 101)
psi0   = basis(2,0) 

H0 = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()
H1 = A/2.0 * sigmax()
args = {'w': omega}
H = [H0, [H1, lambda t,args: sin(args['w'] * t)]]

# noise power spectrum
gamma1 = 0.1
def noise_spectrum(omega):
    return 0.5 * gamma1 * omega/(2*pi)

# find the floquet modes for the time-dependent hamiltonian        
f_modes_0,f_energies = floquet_modes(H, T, args)

# precalculate mode table
f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, linspace(0, T, 500+1), H, T, args) 

# solve the floquet-markov master equation
rho_list = fmmesolve(H, psi0, tlist, [sigmax()], [], [noise_spectrum], T, args).states

# calculate expectation values in the computational basis
p_ex = zeros(shape(tlist), dtype=complex)
for idx, t in enumerate(tlist):
    f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T) 
    p_ex[idx] = expect(num(2), rho_list[idx].transform(f_modes_t, False)) 

# For reference: calculate the same thing with mesolve
p_ex_ref = mesolve(H, psi0, tlist, [sqrt(gamma1) *sigmax()], [num(2)], args).expect[0]

# plot the results
from pylab import *
plot(tlist, real(p_ex),     'ro', tlist, 1-real(p_ex),     'bo')
plot(tlist, real(p_ex_ref), 'r',  tlist, 1-real(p_ex_ref), 'b')
xlabel('Time')
ylabel('Occupation probability')
legend(("Floquet $P_1$", "Floquet $P_0$", "Lindblad $P_1$", "Lindblad $P_0$"))
show()
