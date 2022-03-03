__all__ = ['qpt_plot', 'qpt_plot_combined', 'qpt']

from qutip.tensor import tensor
from qutip.superoperator import spre, spost, mat2vec, vec2mat
from numpy import hstack, real, imag
import scipy.linalg as la
from qutip.visualization import matrix_histogram, matrix_histogram_complex

try:
    import matplotlib.pyplot as plt
except:
    pass


def _index_permutations(size_list, perm=[]):
    """
    Generate a list with all index permutations.

    Parameters
    ----------
    size_list : list
        A list that contains the sizes for each composite system.
    perm : list
        A list of permutations

    Returns
    -------
    perm_idx : list
        List containing index permutations.


    """
    if len(size_list) == 0:
        yield perm
    else:
        for n in range(size_list[0]):
            for ip in _index_permutations(size_list[1:], perm + [n]):
                yield ip


def qpt_plot(chi, lbls_list, title=None, fig=None, axes=None):
    """
    Visualize the quantum process tomography chi matrix. Plot the real and
    imaginary parts separately.

    Parameters
    ----------
    chi : array
        Input QPT chi matrix.
    lbls_list : list
        List of labels for QPT plot axes.
    title : string
        Plot title.
    fig : figure instance
        User defined figure instance used for generating QPT plot.
    axes : list of figure axis instance
        User defined figure axis instance (list of two axes) used for
        generating QPT plot.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    """

    if axes is None or len(axes) != 2:
        if fig is None:
            fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d', position=[0, 0, 1, 1])
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', position=[0, 0, 1, 1])

        axes = [ax1, ax2]

    xlabels = []
    for inds in _index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]]
                                for k in range(len(lbls_list))]))

    matrix_histogram(real(chi), xlabels, xlabels,
                     title=r"real($\chi$)", limits=[-1, 1], ax=axes[0])
    matrix_histogram(imag(chi), xlabels, xlabels,
                     title=r"imag($\chi$)", limits=[-1, 1], ax=axes[1])

    if title and fig:
        fig.suptitle(title)

    return fig, axes


def qpt_plot_combined(chi, lbls_list, title=None,
                      fig=None, ax=None, figsize=(8, 6),
                      threshold=None):
    """
    Visualize the quantum process tomography chi matrix. Plot bars with
    height and color corresponding to the absolute value and phase,
    respectively.

    Parameters
    ----------
    chi : array
        Input QPT chi matrix.

    lbls_list : list
        List of labels for QPT plot axes.

    title : string
        Plot title.

    fig : figure instance
        User defined figure instance used for generating QPT plot.

    ax : figure axis instance
        User defined figure axis instance used for generating QPT plot
        (alternative to the fig argument).

    threshold: float (None)
        Threshold for when bars of smaller height should be transparent. If
        not set, all bars are colored according to the color map.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d', position=[0, 0, 1, 1])

    xlabels = []
    for inds in _index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join(
            [lbls_list[k][inds[k]] for k in range(len(lbls_list))]))

    if not title:
        title = r"$\chi$"

    matrix_histogram_complex(chi, xlabels, xlabels, title=title, ax=ax,
                             threshold=threshold)

    return fig, ax


def qpt(U, op_basis_list):
    """
    Calculate the quantum process tomography chi matrix for a given (possibly
    nonunitary) transformation matrix U, which transforms a density matrix in
    vector form according to:

        vec(rho) = U * vec(rho0)

        or

        rho = vec2mat(U * mat2vec(rho0))

    U can be calculated for an open quantum system using the QuTiP propagator
    function.

    Parameters
    ----------
    U : Qobj
        Transformation operator. Can be calculated using QuTiP propagator
        function.

    op_basis_list : list
        A list of Qobj's representing the basis states.

    Returns
    -------
    chi : array
        QPT chi matrix

    """

    E_ops = []
    # loop over all index permutations
    for inds in _index_permutations([len(ops) for ops in op_basis_list]):
        # loop over all composite systems
        E_op_list = [op_basis_list[k][inds[k]] for k in range(len(
            op_basis_list))]
        E_ops.append(tensor(E_op_list))

    EE_ops = [spre(E1) * spost(E2.dag()) for E1 in E_ops for E2 in E_ops]

    M = hstack([mat2vec(EE.full()) for EE in EE_ops])

    Uvec = mat2vec(U.full())

    chi_vec = la.solve(M, Uvec)

    return vec2mat(chi_vec)
