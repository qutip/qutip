# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2013 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _node_degrees(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr,
                            int num_rows):
    #define all parameters
    cdef unsigned int ii, jj
    cdef np.ndarray[np.intp_t] degree = np.zeros(num_rows,dtype=int)
    #---------------------
    for ii in range(num_rows):
        degree[ii]=ptr[ii+1]-ptr[ii]
        for jj in range(ptr[ii],ptr[ii+1]):
            if ind[jj]==ii:
                #add one if the diagonal is in row ii
                degree[ii]+=1
                break
    return degree
    

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _breadth_first_search(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr,
                            int num_rows, int seed):
    """
    Does a breath first search (BSF) of a graph in sparse CSR format matrix form
    starting at a given seed node.
    """
    #define all parameters
    cdef unsigned int i, j, ii, jj, N = 1
    cdef unsigned int level_start = 0
    cdef unsigned int level_end   = N
    cdef unsigned int current_level = 1
    cdef np.ndarray[np.intp_t] order = -1*np.ones(num_rows, dtype=int)
    cdef np.ndarray[np.intp_t] level = -1*np.ones(num_rows, dtype=int)
    #---------------------
    level[seed] = 0
    order[0] = seed
    
    while level_start < level_end:
        #for nodes of the last level
        for ii in range(level_start,level_end):
            i = order[ii]    
            #add unvisited neighbors to queue
            for jj in range(ptr[i],ptr[i+1]):
                j = ind[jj]
                if level[j] == -1:
                    order[N] = j
                    level[j] = current_level
                    N+=1
        level_start = level_end
        level_end = N
        current_level+=1
    return order, level


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _rcm(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr, int num_rows):
    """
    Reverse Cuthill-McKee ordering of a sparse csr_matrix.
    """
    # define variables
    cdef unsigned int N=0, seed, level_start, N_old, level_end, temp, zz, i, j, ii, jj, kk
    # setup arrays
    cdef np.ndarray[np.intp_t] order = np.zeros(num_rows, dtype=int)
    cdef np.ndarray[np.intp_t] degree = _node_degrees(ind, ptr, num_rows)
    cdef np.ndarray[np.intp_t] inds = np.argsort(degree)
    cdef np.ndarray[np.intp_t] rev_inds = np.argsort(inds)
    
    for zz in range(num_rows):
        if inds[zz] != -1:
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = 0 
            level_end = N
            while level_start < level_end:
                for ii in range(level_start,level_end):
                    i = order[ii] # node i to consider
                    N_old=N # keeps track of how many untouched nodes are in level
                    # add unvisited neighbors
                    for jj in range(ptr[i],ptr[i+1]):   # nodes connected to node i
                        j = ind[jj]                     # j is node number connected to i
                        if inds[rev_inds[j]] != -1:     # if node has not been touched
                            inds[rev_inds[j]] = -1      # touch node
                            order[N] = j                # add current node to order array
                            N += 1                      # add to touched count
                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(N_old,N-1):
                        temp = order[kk]
                        while degree[order[kk+1]] < degree[order[kk]]:
                            order[kk] = order[kk+1]
                            order[kk+1] = temp
                # set level start and end ranges            
                level_start = level_end
                level_end = N
    # return reveresed order for RCM ordering
    return order[::-1]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _pseudo_peripheral_node(np.ndarray[int, mode="c"] ind, 
                        np.ndarray[int, mode="c"] ptr, int num_rows):
    """
    Find a pseudo peripheral node of a graph represented by a sparse csr_matrix.
    """
    #define all parameters
    cdef unsigned int ii, jj, delta, flag, node, start
    cdef int maxlevel, minlevel, minlastnodesdegree
    cdef np.ndarray[np.intp_t] lastnodes
    cdef np.ndarray[np.intp_t] lastnodesdegree
    cdef np.ndarray[np.intp_t] degree = np.zeros(num_rows,dtype=int)
    #---------------------
    #get degrees of each node (row)
    degree = _node_degrees(ind, ptr, num_rows)
    # select an initial starting node
    start = 0
    #set distance delta=0 & flag
    delta = 0
    flag = 1
    while flag:
        # do a level-set traversal from x
        order, level = _breadth_first_search(ind, ptr, num_rows, start)
        # select node in last level with min degree
        maxlevel = max(level)
        lastnodes = np.where(level == maxlevel)[0]
        lastnodesdegree = degree[lastnodes]
        minlastnodesdegree = min(lastnodesdegree)
        node = np.where(lastnodesdegree == minlastnodesdegree)[0][0]
        node = lastnodes[node]
        # if d(x,y)>delta, set, and do another BFS fro this minimal node
        if level[node] > delta:
            start = node
            delta = level[node]
        else:
            flag = 0
    return start, order, level