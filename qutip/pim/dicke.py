"""
Permutation Invariant Matrix calculation for Dicke state dynamics
"""
from math import factorial
from decimal import Decimal

import numpy as np


def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of
    ensembles with odd number of systems.

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    nds: int
        The number of Dicke states
    """
    if (N < 0) or (not isinstance(N, (np.integer, int))):
        raise ValueError("Number of two level systems should be a positive int")

    nds = (N/2 + 1)**2 - (N % 2)/4
    return int(nds)


def num_two_level(nds):
    """
    The number of two level systems (TLS), given the number of Dicke states.
    Inverse function of num_dicke_states(N)

    Parameters
    ----------
    nds: int
         The number of Dicke states

    Returns
    -------
    N: int
        The number of two level systems
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2 * (np.sqrt(nds) - 1)

    else:
        # N is odd
        N = 2 * (np.sqrt(nds + 1/4) - 1)

    return int(N)


def irreducible_dim(N, j):
    """
    Calculates the dimension of the subspace that accounts for the irreducible
    representations of a given state (j, m) when the degeneracy is removed.

    Parameters
    ----------
    N: int
        The number of two level systems
    j: int
        Total spin eigenvalue

    Returns
    -------
    djn: int
        The irreducible dimension
    """
    num = Decimal(factorial(N))
    den = Decimal(factorial(N/2 - j)) * Decimal(factorial(N/2 + j + 1))

    djn = float(num/den)
    djn = djn * (2*j + 1)

    return (djn)


def num_dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space indexed by
    (j,m), for a collection of N two-level systems. It counts how many
    different "j" exists

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5
    return int(Nj)


def generate_dicke_space(N):
    """
    Generates matrix where elements existing in the Dicke space take the value 1.

    Each of the elements with value 1 in this matrix is a valid Dicke state element
    with a corresponding |j, m> For instance for 6 two level systems, the valid
    elements would be the ones below. We use a check to find if a particular element
    is a valid dicke space element before using it to calculate the matrix governing
    the evolution of the system. This function does not have any specific use other
    than to visualize the dicke space


    N = 6

    1  
    1 1 
    1 1 1
    1 1 1 1
    1 1 1
    1 1 
    1
    
    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    dicke_space: array
        A 2D array with the valid dicke state elements set to 1
    """        
    rows = N + 1
    cols = 0

    if (rows % 2) == 0:
        cols = int((rows/2))

    else:
        cols = int((rows + 1)/2)

    dicke_space = np.zeros((rows, cols), dtype = int)

    for (i, j) in np.ndindex(rows, cols):
        dicke_space[i, j] = self.isdicke(i, j)

    return (dicke_space)


def isdicke(N, index):
    """
    Check if the index of the element specified is a valid dicke space element.

    This depends on how many two level systems you consider. For instance, at
    N = 6, the dicke space would look like the following:

    1  
    1 1 
    1 1 1
    1 1 1 1
    1 1 1
    1 1 
    1
    
    Thus (0, 0), (1, 0), (2, 2) etc are valid dicke space elements but 
    (0, 1), (1, 3) do not exist

    Parameters
    ----------
    index: tuple
        Index of the element (i, j) in Dicke space which needs to be checked
    """
    # The number of rows is N + 1
    dicke_row, dicke_col = index
    rows = N + 1
    cols = 0

    # The number of columns depends on whether N is even or odd.
    if (N % 2) == 0:
        cols = int(N/2 + 1)
    else:
        cols = int(N/2 + 1/2)

    # Check if the given indices falls inside the matrix region
    if (dicke_row > rows) or (dicke_row < 0):
        return (False)

    if (dicke_col > cols) or (dicke_col < 0):
        return (False)

    # If the element is in the upper region, it should lie below the diagonal
    if (dicke_row < int(rows/2)) and (dicke_col > dicke_row):
        return False

    # If the element is in the lower region, it should lie above the diagonal
    if (dicke_row >= int(rows/2)) and (rows - dicke_row <= dicke_col):
        return False
    
    else:
        return True


def get_j_m(N, index):
    """
    Get the value of j and m for the particular Dicke space element given N TLS

    For N = 6 |j, m> would be :

    | 3, 3>
    | 3, 2> | 2, 2>
    | 3, 1> | 2, 1> | 1, 1>
    | 3, 0> | 2, 0> | 1, 0> |0, 0>
    | 3,-1> | 2,-1> | 1,-1>
    | 3,-2> | 2,-2>
    | 3,-3>

    Parameters
    ----------
    index: tuple
        Index of the element (i, j) in Dicke space

    Returns
    -------
    j, m: float
        The j and m values for this element
    """
    if not isdicke(N, index):
        return False

    else:
        dicke_row, dicke_col = index
        j = N/2 - dicke_col
        m = N/2 - dicke_row

        return(j, m)

def is_j_m(N, jm):
    """
    Test if |j, m> is valid for the given number of two level systems

    Parameters
    ----------
    N: int
        Number of two level systems

    jm: tuple of floats
        A tuple of (j, m) values as type float
    """
    j, m = jm

    dicke_row = int(N/2 - m)
    dicke_col = int(N/2 - j)

    if isdicke(N, (dicke_row, dicke_col)):
        return(True)

    else:
        return(False)

def get_k(N, index):
    """
    Get the index `k` in the dicke state vector corresponding to given
    index (row, col) entry where the entry is non-zero. This also forms
    the row index in the matrix M for the given dicke state element

    For N = 6, the dice space will look like this

    1
    1 1
    x 1 1
    1 1 1 1
    1 y 1
    1 1
    1

    The elements marked `x` and `y` here, will have the `k` value as
    2 and 10 respectively (indexing starts from 0) as they are the 3rd and 11th
    element in the dicke ladder.

    Parameters
    ----------
    index: tuple
        Index of the element (i, j) in Dicke space

    Returns
    -------
    k: int
        The index k for the non zero element in the dicke state vector
    """
    dicke_row, dicke_col = index

    if not isdicke(N, index):
        return (False)

    if dicke_row == 0:
        k = dicke_col

    else:
        k = int(((dicke_col)/2) * (2 * (N + 1) - 2 * (dicke_col - 1)) + (dicke_row - (dicke_col)))

    return k

def initial_dicke_state(N, jm0):
    """
    Generates a initial dicke state vector given the number of two-level systems, from specified |j0, m0>

    Parameters
    ----------
    N: int
        The number of two level systems

    jm0: tuple
        A tuple containing (j0, m0) value as type float

    Returns
    -------
    rho: array
        The initial dicke state vector constructed from this j0, m0 value
    """
    j, m = jm0
    nds = num_dicke_states(N)
    rho0 = np.zeros(nds)

    dicke_row = int(N/2 - m)
    dicke_col = int(N/2 - j)

    k = get_k(N, (dicke_row, dicke_col))

    if k is False:
        raise ValueError("Invalid value of |j0, m0> for given number of TLS")

    else:
        rho0[k] = 1.
        return(rho0)
