#
# Plots the entangled superposition
# 3-qubit GHZ eigenstate |up,up,up> + |dn,dn,dn>
#
# From the xGHZ qotoolbox example by Sze M. Tan
#
from qutip import *
from pylab import *


def run():
    # create spin operators for the three qubits.
    sx1 = tensor(sigmax(), qeye(2), qeye(2))
    sy1 = tensor(sigmay(), qeye(2), qeye(2))

    sx2 = tensor(qeye(2), sigmax(), qeye(2))
    sy2 = tensor(qeye(2), sigmay(), qeye(2))

    sx3 = tensor(qeye(2), qeye(2), sigmax())
    sy3 = tensor(qeye(2), qeye(2), sigmay())

    # Calculate products
    op1 = sx1 * sy2 * sy3
    op2 = sy1 * sx2 * sy3
    op3 = sy1 * sy2 * sx3
    opghz = sx1 * sx2 * sx3

    # Find simultaneous eigenkets of op1,op2,op3 and opghz
    evalues, states = simdiag([op1, op2, op3, opghz])

    # convert last eigenstate to density matrix
    rho0 = ket2dm(states[-1])
    # create labels for density matrix plot
    upupup = "$|\\uparrow,\\uparrow,\\uparrow\\rangle$"
    dndndn = "$|\\downarrow,\\downarrow,\\downarrow\\rangle$"
    title = "3-Qubit GHZ state: $\\frac{1}{\\sqrt{2}}$" + upupup + "+" + dndndn
    xlabels = [""] * 8
    xlabels[0] = upupup  # set first xaxes label
    xlabels[-1] = dndndn  # set last xaxes label
    ylabels = [""] * 8
    ylabels[-1] = upupup  # set last yaxes label
    # generate plot with labels
    matrix_histogram(rho0, xlabels=xlabels, ylabels=ylabels, title=title)
    show()
    close()

if __name__ == '__main__':
    run()
