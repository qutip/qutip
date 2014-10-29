# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:54:59 2014

@author: Alexander Pitchford
"""

import numpy as np

def qft(nQubits):
    """
    Returns array that represents Quantum Fourier Transformation matrix
    with 2^n rows and coulmns
    """
    n = 2**nQubits
    # Compute nth root of unity
    w = np.exp(2*np.pi*1j/n)
    Powers = np.empty([n, n], dtype=int)
    # Computer matrix of powers to raise w to
    Row = np.arange(0, n)
    for r in range(n):
        Powers[r, :] = Row*r
    #print "Powers:"
    #print Powers
    # use Power and w to create the QFT transformation matrix
    QftMat = np.empty([n, n], dtype=complex)
    QftMat = w**Powers
    QftMat = QftMat/np.sqrt(n)
    #print QftMat
    return QftMat
