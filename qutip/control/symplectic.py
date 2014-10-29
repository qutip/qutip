# -*- coding: utf-8 -*-
"""
Created on Mon Jul 07 21:07:03 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Utility functions for symplectic matrices
"""

import numpy as np

def calc_omega(n):
    """
    Calculate the 2n x 2n omega matrix
    Used in calcualating the propagators in systems described by symplectic
    matrices
    returns omega
    """
    
    omg = np.zeros((2*n, 2*n))
    for j in range(2*n):
        for k in range(2*n):
            if (k == j+1):
                omg[j, k] = (1 + (-1)**j)/2
            if (k == j-1):
                omg[j, k] = -(1 - (-1)**j)/2
                
    return omg
    
    