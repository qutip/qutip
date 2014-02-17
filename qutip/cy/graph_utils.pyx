# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without 
#    modification, are permitted provided that the following conditions are 
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _node_degrees(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr,
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
def _breadth_first_search(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr,
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
def _rcm(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr, int num_rows):
    """
    Reverse Cuthill-McKee ordering of a sparse csr or csc matrix.
    """
    # define variables
    cdef unsigned int N=0, N_old, seed, level_start, level_end, temp, temp2, zz, i, j, ii, jj, kk, ll
    # setup arrays
    cdef np.ndarray[np.intp_t] order = np.zeros(num_rows, dtype=int)
    cdef np.ndarray[np.intp_t] degree = _node_degrees(ind, ptr, num_rows)
    cdef np.ndarray[np.intp_t] inds = np.argsort(degree)
    cdef np.ndarray[np.intp_t] rev_inds = np.argsort(inds)
    cdef np.ndarray[np.intp_t] temp_degrees = np.zeros(num_rows, dtype=int)
    #loop over zz takes into account possible disconnected graph.
    for zz in range(num_rows):
        if inds[zz] != -1: # Do BFS with seed=inds[zz]
            seed = inds[zz] # seed node for BFS
            order[N] = seed # add seed to order
            N += 1          # increase # touched nodes
            inds[rev_inds[seed]] = -1 #mark touched node inds
            level_start = N-1 
            level_end = N
            while level_start < level_end:
                for ii in range(level_start,level_end):
                    i = order[ii] # node i to consider
                    N_old=N # old # of touched nodes
                
                    # add unvisited neighbors
                    for jj in range(ptr[i],ptr[i+1]):   # nodes connected to node i
                        j = ind[jj] # j is node number connected to i
                        if inds[rev_inds[j]] != -1:     # if node not touched
                            inds[rev_inds[j]] = -1      # touch node
                            order[N] = j                # add node to order
                            N += 1                      # add to touched count
                    #Add values to temp_degrees array for insertion sort
                    temp=0
                    for kk in range(N_old,N):
                        temp_degrees[temp]=degree[order[kk]]
                        temp+=1
                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(1,(N-N_old)):
                        temp = order[kk]
                        temp2 = temp_degrees[kk]
                        ll = kk - 1
                        while (ll>=0) and (temp_degrees[ll] > temp2):
                            temp_degrees[ll+1] = temp_degrees[ll]
                            order[ll+1] = order[ll]
                            ll-=1
                        order[ll+1] = temp
                        temp_degrees[ll+1] = temp2
            
                # set next level start and end ranges            
                level_start = level_end
                level_end = N
        if N==num_rows:
            break
    # return reveresed order for RCM ordering
    return order[::-1]


@cython.boundscheck(False)
@cython.wraparound(False)
def _pseudo_peripheral_node(np.ndarray[int, mode="c"] ind, 
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
        # if d(x,y)>delta, set, and do another BFS from this minimal node
        if level[node] > delta:
            start = node
            delta = level[node]
        else:
            flag = 0
    return start, order, level


@cython.boundscheck(False)
@cython.wraparound(False)
def _bfs_matching(np.ndarray[int, mode="c"] inds, np.ndarray[int, mode="c"] ptrs, int n):
    cdef np.ndarray[np.intp_t] visited = np.zeros(n,dtype=int)     #visited array
    cdef np.ndarray[np.intp_t] queue = np.zeros(n,dtype=int)       #queue array
    cdef np.ndarray[np.intp_t] previous = np.zeros(n,dtype=int)    #prev visited array
    cdef np.ndarray[np.intp_t] match = -1*np.ones(n,dtype=int)     #returned matching
    cdef np.ndarray[np.intp_t] row_match = -1*np.ones(n,dtype=int) #row_matching
    cdef int queue_ptr, queue_col, ptr, next_num, i, j, queue_size, row, col, temp, eptr
    next_num=1 
    for i in range(n):
        if (match[i] == -1 and (ptrs[i] != ptrs[i+1])):
            queue[0] = i
            queue_ptr = 0
            queue_size = 1
            while (queue_size > queue_ptr):
                queue_col = queue[queue_ptr]
                queue_ptr+=1
                eptr = ptrs[queue_col + 1]
                for ptr in range(ptrs[queue_col], eptr):
                    row = inds[ptr]
                    temp = visited[row]
                    if (temp != next_num and temp != -1):
                        previous[row] = queue_col
                        visited[row] = next_num
                        col = row_match[row]
                        if (col == -1):
                            while (row != -1):
                                col = previous[row]
                                temp = match[col]
                                match[col] = row 
                                row_match[row] = col
                                row = temp
                            next_num+=1
                            queue_size = 0
                            break
                        else:
                            queue[queue_size] = col
                            queue_size+=1
            if (match[i] == -1):
                for j in range(1,queue_size):
                    visited[match[queue[j]]] = -1
    return match

@cython.boundscheck(False)
@cython.wraparound(False)
def _max_row_weights(np.ndarray[DTYPE_t, mode="c"] data, np.ndarray[int, mode="c"] inds, np.ndarray[int, mode="c"] ptrs, int ncols):
    """
    Finds the largest abs value in each matrix column and the max. total number of elements
    in the cols (given by weights[-1]).
    
    Here we assume that the user already took the ABS value of the data.  This keeps us from
    having to call abs over and over.
    
    """
    #define all parameters
    cdef np.ndarray[DTYPE_t] weights = np.zeros(ncols+1,dtype=float)
    cdef int ln, mx, ii, jj
    cdef DTYPE_t weight, current
    #---------------------
    mx=0
    for jj in range(ncols):
        ln=(ptrs[jj+1]-ptrs[jj])
        if ln>mx:
            mx=ln
        #weight=np.abs(data[ptrs[jj]])
        weight=data[ptrs[jj]]
        for ii in range(ptrs[jj]+1,ptrs[jj+1]):
            #current=np.abs(data[ii])
            current=data[ii]
            if current>weight:
                weight=current
        weights[jj]=weight
    weights[ncols]=mx
    return weights


@cython.boundscheck(False)
@cython.wraparound(False)
def _weighted_bfs_matching(np.ndarray[DTYPE_t, mode="c"] data, np.ndarray[int, mode="c"] inds, np.ndarray[int, mode="c"] ptrs, int n):
    """
    Here we assume that the user already took the ABS value of the data.  This keeps us from
    having to call abs over and over.
    """
    #define all parameters
    cdef np.ndarray[np.intp_t] visited = np.zeros(n,dtype=int)     #visited array
    cdef np.ndarray[np.intp_t] queue = np.zeros(n,dtype=int)       #queue array
    cdef np.ndarray[np.intp_t] previous = np.zeros(n,dtype=int)    #prev visited array
    cdef np.ndarray[np.intp_t] match = -1*np.ones(n,dtype=int)     #returned matching
    cdef np.ndarray[np.intp_t] row_match = -1*np.ones(n,dtype=int) #row_matching
    cdef np.ndarray[DTYPE_t] weights = _max_row_weights(data, inds, ptrs, n) #max weights in each column
    cdef np.ndarray[np.intp_t] order = np.argsort(-weights[0:n])    #order in which cols traversed
    cdef np.ndarray[np.intp_t] row_order = np.zeros(weights[n],dtype=int)   #temp array to order row
    cdef np.ndarray[DTYPE_t] temp_weights = np.zeros(weights[n],dtype=float)   #temp array for row weights
    cdef int queue_ptr, queue_col, queue_size, next_num, i, j, zz, ll, kk, row, col, temp, eptr, temp2
    #---------------------
    next_num=1 
    for i in range(n):
        zz=order[i] #cols with largest abs values first
        if (match[zz] == -1 and (ptrs[zz] != ptrs[zz+1])):
            queue[0] = zz
            queue_ptr = 0
            queue_size = 1
            while (queue_size > queue_ptr):
                queue_col = queue[queue_ptr]
                queue_ptr+=1
                eptr = ptrs[queue_col + 1]
                #get row inds in current column
                temp=ptrs[queue_col]
                for kk in range(eptr-ptrs[queue_col]):
                    row_order[kk]=inds[temp]
                    temp_weights[kk]=data[temp]
                    temp+=1
                #linear sort by row weight
                for kk in range(1,(eptr-ptrs[queue_col])):
                    val = temp_weights[kk]
                    row_val = row_order[kk]
                    ll = kk - 1
                    while (ll>=0) and (temp_weights[ll] > val):
                        temp_weights[ll+1] = temp_weights[ll]
                        row_order[ll+1] = row_order[ll]
                        ll-=1
                    temp_weights[ll+1] = val
                    row_order[ll+1] = row_val
                #go through rows by decending weight
                temp2=(eptr-ptrs[queue_col])-1
                for kk in range(eptr-ptrs[queue_col]):
                    row = row_order[temp2-kk]
                    temp = visited[row]
                    if (temp != next_num and temp != -1):
                        previous[row] = queue_col
                        visited[row] = next_num
                        col = row_match[row]
                        if (col == -1):
                            while (row != -1):
                                col = previous[row]
                                temp = match[col]
                                match[col] = row 
                                row_match[row] = col
                                row = temp
                            next_num+=1
                            queue_size = 0
                            break
                        else:
                            queue[queue_size] = col
                            queue_size+=1
            if (match[zz] == -1):
                for j in range(1,queue_size):
                    visited[match[queue[j]]] = -1
    return match
