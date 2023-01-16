# -*- coding: utf-8 -*-
# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Utility functions for symplectic matrices
"""

import numpy as np


def calc_omega(n):
    """
    Calculate the 2n x 2n Omega matrix
    Used as dynamics generator phase to calculate symplectic propagators

    Parameters
    ----------
    n : scalar(int)
        number of modes in oscillator system

    Returns
    -------
    array(float)
        Symplectic phase Omega
    """

    omg = np.zeros((2*n, 2*n))
    for j in range(2*n):
        for k in range(2*n):
            if k == j+1:
                omg[j, k] = (1 + (-1)**j)/2
            if k == j-1:
                omg[j, k] = -(1 - (-1)**j)/2

    return omg
