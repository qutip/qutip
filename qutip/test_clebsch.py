# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:10:42 2020

@author: twalk
"""

#returns True twice if clebsch is working for both int and float inputs

import qutip as qu

cgcoeff1 = qu.clebsch(0.5, 1, 0.5, 0.5, -1, -0.5) #should be sqrt(2/3)

print(round(cgcoeff1, 14)  == round((2/3)**0.5, 14)) #round to account for error in final digit

cgcoeff2 = qu.clebsch(1, 1, 1, 1, -1, 0) #should be sqrt(1/2)

print(round(cgcoeff2, 14) == round(1/2**0.5, 14)) 
