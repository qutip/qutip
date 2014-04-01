#
# 3D Wigner and Q-functions for
# a squeezed coherent state.
#
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
from qutip import *

def run():
    # setup constants:
    N = 20
    alpha = -1.0  # Coherent amplitude of field
    epsilon = 0.5j  # Squeezing parameter

    D = displace(N, alpha)     # Displacement
    S = squeeze(N, epsilon)     # Squeezing
    psi = D * S * basis(N, 0)  # Apply to vacuum state

    xvec = linspace(-6, 6, 150)
    X, Y = meshgrid(xvec, xvec)
    W = wigner(psi, xvec, xvec)

    Q = qfunc(psi, xvec, xvec)
    fig = figure()
    ax = Axes3D(fig, azim=-62, elev=25)
    ax.plot_surface(X, Y, W, rstride=2, cstride=2, cmap=cm.jet, lw=.1)
    ax.set_xlim3d(-6, 6)
    ax.set_xlim3d(-6, 6)
    ax.set_zlim3d(0, 0.4)
    title('Wigner function of squeezed coherent state')

    fig = figure()
    ax2 = Axes3D(fig, azim=-43, elev=37)
    ax2.plot_surface(X, Y, Q, rstride=2, cstride=2, cmap=cm.jet, lw=.1)
    ax2.set_xlim3d(-6, 6)
    ax2.set_xlim3d(-6, 6)
    ax2.set_zlim3d(0, 0.2)
    title('Q-function of squeezed coherent state')
    show()
    close('all')


if __name__ == "__main__":
    run()
