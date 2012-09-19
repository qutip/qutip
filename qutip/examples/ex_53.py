#
# Plot the process tomography matrix for a 2-qubit iSWAP gate.
#

from qutip import *

def run():
    g = 1.0 * 2 * pi # coupling strength
    g1 = 0.75        # relaxation rate
    g2 = 0.25        # dephasing rate
    n_th = 1.5       # bath temperature

    T = pi/(4*g)
    H = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
    psi0 = tensor(basis(2,1), basis(2,0))

    c_ops = []
    # qubit 1 collapse operators
    sm1 = tensor(sigmam(), qeye(2))
    sz1 = tensor(sigmaz(), qeye(2))
    c_ops.append(sqrt(g1 * (1+n_th)) * sm1)
    c_ops.append(sqrt(g1 * n_th) * sm1.dag())
    c_ops.append(sqrt(g2) * sz1)
    # qubit 2 collapse operators
    sm2 = tensor(qeye(2), sigmam())
    sz2 = tensor(qeye(2), sigmaz())
    c_ops.append(sqrt(g1 * (1+n_th)) * sm2)
    c_ops.append(sqrt(g1 * n_th) * sm2.dag())
    c_ops.append(sqrt(g2) * sz2)

    # process tomography basis
    op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * 2
    op_label = [["i", "x", "y", "z"]] * 2

    # dissipative gate
    U_diss = propagator(H, T, c_ops)
    chi = qpt(U_diss, op_basis)
    qpt_plot_combined(chi, op_label)

    # ideal gate 
    U_psi = (-1j * H * T).expm()
    U_ideal = spre(U_psi) * spost(U_psi.dag())
    chi = qpt(U_ideal, op_basis)
    qpt_plot_combined(chi, op_label)

    show()

if __name__=='__main__':
    run()
