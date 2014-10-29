# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:14:31 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Some utility functions for the library
"""

def write_array_to_file(a, fname='array.txt', dtype=float):
    with open(fname, 'w') as f:
        write_array(a, f, dtype=dtype)
    
def write_array(a, f, dtype=float):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (dtype == complex):
                f.write("{:17.4f}".format(a[i, j]))
            elif (dtype == float):
                f.write("{:11.4f}".format(a[i, j]))
            elif (dtype == int):
                f.write("{:6n}".format(a[i, j]))
            else:
                f.write("{}".format(a[i, j]))
            if (j < a.shape[1] - 1):
                f.write(",")
                
        f.write("\n")