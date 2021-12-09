__all__ = ['berry_curvature', 'plot_berry_curvature']

from qutip import (Qobj, tensor, basis, qeye, isherm, sigmax, sigmay, sigmaz)
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass


def berry_curvature(eigfs):
    """Computes the discretized Berry curvature on the two dimensional grid
    of parameters. The function works well for cases with no band mixing.

    Parameters
    ==========
    eigfs : numpy ndarray
        4 dimensional numpy ndarray where the first two indices are for the two
        discrete values of the two parameters and the third is the index of the
        occupied bands. The fourth dimension holds the eigenfunctions.

    Returns
    -------
    b_curv : numpy ndarray
        A two dimensional array of the discretized Berry curvature defined for
        the values of the two parameters defined in the eigfs.
    """
    nparam0 = eigfs.shape[0]
    nparam1 = eigfs.shape[1]
    nocc = eigfs.shape[2]
    b_curv = np.zeros((nparam0-1, nparam1-1), dtype=float)

    for i in range(nparam0-1):
        for j in range(nparam1-1):
            rect_prd = np.identity(nocc, dtype=complex)
            innP0 = np.zeros([nocc, nocc], dtype=complex)
            innP1 = np.zeros([nocc, nocc], dtype=complex)
            innP2 = np.zeros([nocc, nocc], dtype=complex)
            innP3 = np.zeros([nocc, nocc], dtype=complex)

            for k in range(nocc):
                for l in range(nocc):
                    wf0 = eigfs[i, j, k, :]
                    wf1 = eigfs[i+1, j, l, :]
                    innP0[k, l] = np.dot(wf0.conjugate(), wf1)

                    wf1 = eigfs[i+1, j, k, :]
                    wf2 = eigfs[i+1, j+1, l, :]
                    innP1[k, l] = np.dot(wf1.conjugate(), wf2)

                    wf2 = eigfs[i+1, j+1, k, :]
                    wf3 = eigfs[i, j+1, l, :]
                    innP2[k, l] = np.dot(wf2.conjugate(), wf3)

                    wf3 = eigfs[i, j+1, k, :]
                    wf0 = eigfs[i, j, l, :]
                    innP3[k, l] = np.dot(wf3.conjugate(), wf0)

            rect_prd = np.dot(rect_prd, innP0)
            rect_prd = np.dot(rect_prd, innP1)
            rect_prd = np.dot(rect_prd, innP2)
            rect_prd = np.dot(rect_prd, innP3)

            dett = np.linalg.det(rect_prd)
            curl_z = np.angle(dett)
            b_curv[i, j] = curl_z

    return b_curv


def plot_berry_curvature(eigfs):
    """Plots the discretized Berry curvature on the two dimensional grid
    of parameters. The function works well for cases with no band mixing."""
    b_curv = berry_curvature(eigfs)
    fig, ax = plt.subplots()
    ax.imshow(b_curv, origin="lower")
    ax.set_title("Berry curvature")
    ax.set_xlabel(r"$Parameter0$")
    ax.set_ylabel(r"$Parameter1$")
    fig.tight_layout()
    fig.savefig("berry_curvature.pdf")
