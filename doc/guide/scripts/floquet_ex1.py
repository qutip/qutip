from qutip import *

delta = 0.2 * 2*pi; eps0  = 1.0 * 2*pi
A     = 0.5 * 2*pi; omega = 1.0 * 2*pi
T      = (2*pi)/omega 
tlist  = linspace(0.0, 10 * T, 101)
psi0   = basis(2,0) 

H0 = - delta/2.0 * sigmax() - eps0/2.0 * sigmaz()
H1 = A/2.0 * sigmaz()
args = {'w': omega}
H = [H0, [H1, lambda t,args: sin(args['w'] * t)]]

# find the floquet modes for the time-dependent hamiltonian        
f_modes_0,f_energies = floquet_modes(H, T, args)

# decompose the inital state in the floquet modes
f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
       
# calculate the wavefunctions using the from the floquet modes
p_ex = zeros(len(tlist))  
for n, t in enumerate(tlist):
    psi_t = floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args)
    p_ex[n] = expect(num(2), psi_t)

# For reference: calculate the same thing with mesolve
p_ex_ref = mesolve(H, psi0, tlist, [], [num(2)], args).expect[0]

# plot the results
from pylab import *
plot(tlist, real(p_ex),     'ro', tlist, 1-real(p_ex),     'bo')
plot(tlist, real(p_ex_ref), 'r',  tlist, 1-real(p_ex_ref), 'b')
xlabel('Time')
ylabel('Occupation probability')
legend(("Floquet $P_1$", "Floquet $P_0$", "Lindblad $P_1$", "Lindblad $P_0$"))
show()
