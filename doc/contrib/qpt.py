from qutip.tensor import tensor
from qutip.superoperator import spre, spost, mat2vec, vec2mat
from qutip.Qobj import Qobj
from numpy import hstack
import scipy.linalg as la

from pylab import *

def index_permutations(size_list, perm=[]):
    """
    Generate a list with all index permutations. size_list is a list that 
    contains the sizes for each composite system.
    """
    if len(size_list) == 0:
        yield perm
    else:
        for n in range(size_list[0]):
            for ip in index_permutations(size_list[1:], perm + [n]):
                yield ip

def qpt_plot(chi, lbls_list, title=None, fig=None):
    """
    Visualize the quantum process tomography chi matrix. Plot the real and
    imaginary parts separately.
    """
    if fig == None:
        fig = figure(figsize=(16,8))

    xlabels = []
    for inds in index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]] for k in range(len(lbls_list))]))        

    ax = fig.add_subplot(1,2,1, projection='3d', position=[0, 0, 1, 1])
    matrix_histogram(real(chi), xlabels, xlabels, r"real($\chi$)", [-1,1], ax)

    ax = fig.add_subplot(1,2,2, projection='3d', position=[0, 0, 1, 1])
    matrix_histogram(imag(chi), xlabels, xlabels, r"imag($\chi$)", [-1,1], ax)

    if title:
        fig.suptitle(title)

def qpt_plot_combined(chi, lbls_list, title=None, fig=None):
    """
    Visualize the quantum process tomography chi matrix. Plot bars with
    height that correspond to the absolute value and color that correspond
    to the phase.
    """
    if fig == None:
        fig = figure(figsize=(8,6))

    xlabels = []
    for inds in index_permutations([len(lbls) for lbls in lbls_list]):
        xlabels.append("".join([lbls_list[k][inds[k]] for k in range(len(lbls_list))]))        

    if not title:
        title = r"$\chi$"

    ax = fig.add_subplot(1,1,1, projection='3d', position=[0, 0, 1, 1])

    matrix_histogram_complex(chi, xlabels, xlabels, title, None, ax)

def qpt(U, op_basis_list):
    """
    Calculate the quantum process tomography chi matrix for a given 
    (possibly nonunitary) transformation matrix U, which transforms a 
    density matrix in vector form according to:

        vec(rho) = U * vec(rho0)

        or

        rho = vec2mat(U * mat2vec(rho0))

    U can be calculated for an open quantum system using the QuTiP propagator
    function.
    """

    E_ops = []
    # loop over all index permutations
    for inds in index_permutations([len(op_list) for op_list in op_basis_list]):
        # loop over all composite systems
        E_op_list = [op_basis_list[k][inds[k]] for k in range(len(op_basis_list))]
        E_ops.append(tensor(E_op_list))

    EE_ops = [spre(E1) * spost(E2.dag()) for E1 in E_ops for E2 in E_ops]

    M = hstack([mat2vec(EE.full()) for EE in EE_ops])

    Uvec = mat2vec(U.full())

    chi_vec = la.solve(M, Uvec)

    return vec2mat(chi_vec)

def matrix_histogram(M, xlabels, ylabels, title, limits=None, ax=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.
    
    Returns
    -------

        An matplotlib axes instance for the plot.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n=size(M) 
    xpos,ypos=meshgrid(range(M.shape[0]),range(M.shape[1]))
    xpos=xpos.T.flatten()-0.5 
    ypos=ypos.T.flatten()-0.5 
    zpos = zeros(n) 
    dx = dy = 0.8 * ones(n) 
    dz = real(M.flatten()) 
    
    if limits: # check that limits is a list type
        z_min = limits[0]
        z_max = limits[1]
    else:
        z_min = min(dz)
        z_max = max(dz)
        
    norm=mpl.colors.Normalize(z_min, z_max) 
    cmap=get_cmap('jet') # Spectral
    colors=cmap(norm(dz))

    if ax == None:
        fig = plt.figure()
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    plt.title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(IndexLocator(1,-0.5))
    ax.set_xticklabels(xlabels) 
    ax.tick_params(axis='x', labelsize=14)

    # y axis
    ax.axes.w_yaxis.set_major_locator(IndexLocator(1,-0.5)) 
    ax.set_yticklabels(ylabels) 
    ax.tick_params(axis='y', labelsize=14)

    # z axis
    ax.axes.w_zaxis.set_major_locator(IndexLocator(1,0.5))
    ax.set_zlim3d([z_min, z_max])

    # color axis
    cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    return ax

def matrix_histogram_complex(M, xlabels, ylabels, title, limits=None, ax=None):
    """
    Draw a histogram for the amplitudes of matrix M, using the argument of each element
    for coloring the bars, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.
    
    Returns
    -------

        An matplotlib axes instance for the plot.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n=size(M) 
    xpos,ypos=meshgrid(range(M.shape[0]),range(M.shape[1]))
    xpos=xpos.T.flatten()-0.5 
    ypos=ypos.T.flatten()-0.5 
    zpos = zeros(n) 
    dx = dy = 0.8 * ones(n) 
    Mvec = M.flatten()
    dz = abs(Mvec) 
    
    # make small numbers real, to avoid random colors
    idx, = where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    if limits: # check that limits is a list type
        phase_min = limits[0]
        phase_max = limits[1]
    else:
        phase_min = -pi
        phase_max = pi
        
    norm=mpl.colors.Normalize(phase_min, phase_max) 

    # create a cyclic colormap
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
            'green': ((0.00, 0.0, 0.0),
                      (0.25, 1.0, 1.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
            'red':   ((0.00, 1.0, 1.0),
                      (0.25, 0.5, 0.5),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 1.0, 1.0))}
    cmap = matplotlib.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)

    colors = cmap(norm(angle(Mvec)))

    if ax == None:
        fig = plt.figure()
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    plt.title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(IndexLocator(1,-0.5))
    ax.set_xticklabels(xlabels) 
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ax.axes.w_yaxis.set_major_locator(IndexLocator(1,-0.5)) 
    ax.set_yticklabels(ylabels) 
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    #ax.axes.w_zaxis.set_major_locator(IndexLocator(1,0.5))
    ax.set_zlim3d([0, 1])
    #ax.set_zlabel('abs')

    # color axis
    cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_ticks([-pi, -pi/2, 0, pi/2, pi])
    cb.set_ticklabels((r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'))
    cb.set_label('arg')

    return ax

def iswap():
    """Quantum object representing the iSWAP gate.
    
    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate
    
    Examples
    --------
    >>> iswap()
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    return Qobj(array([[1,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,1]]), dims=[[2, 2], [2, 2]])


def sqrtiswap():
    """Quantum object representing the square root iSWAP gate.
    
    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate
    
    Examples
    --------
    >>> sqrtiswap()
    Quantum object: dims = [[2, 2], [2, 2]], shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j   0.00000000+0.j          0.00000000+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.70710678+0.j          0.00000000-0.70710678j  0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000-0.70710678j  0.70710678+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000+0.j          0.00000000+0.j          1.00000000+0.j]]    
    """
    return Qobj(array([[1,0,0,0], [0, 1/sqrt(2), -1j/sqrt(2), 0], [0, -1j/sqrt(2), 1/sqrt(2), 0], [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]])
