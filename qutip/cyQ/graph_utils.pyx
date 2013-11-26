import numpy as np
cimport numpy as np
cimport cython
ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _breadth_first_search(np.ndarray[int, mode="c"] ind, np.ndarray[int, mode="c"] ptr,
                            int seed, np.ndarray[np.intp_t, mode="c"] level, 
                            np.ndarray[np.intp_t, mode="c"] order):
    """
    Does a breath first search (BSF) of a graph in sparse CSR format matrix 
    starting at a given seed node.
    """
    #define all parameters
    cdef unsigned int i, j, ii, jj, N = 1
    cdef unsigned int level_start = 0
    cdef unsigned int level_end   = N
    cdef unsigned int current_level = 1
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
    
    return level, order


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _pseudo_peripheral_node(np.ndarray[int, mode="c"] ind, 
                        np.ndarray[int, mode="c"] ptr, int num_rows):
    """
    Find a pseudo peripheral node of a graph represented by a sparse csr_matrix.
    """
    #define all parameters
    cdef unsigned int delta, flag, node, x
    cdef int maxlevel, minlevel, minlastnodesvalence
    cdef np.ndarray[np.intp_t] order = np.empty(num_rows, dtype=int)
    cdef np.ndarray[np.intp_t] level = -1*np.ones(num_rows, dtype=int)
    cdef np.ndarray[np.intp_t] lastnodes
    cdef np.ndarray[int] valence, lastnodesvalence
    
    #---------------------
    valence = np.diff(ind)
    # select an initial node x
    x = int(np.random.rand() * num_rows)
    #det delta=0
    delta = 0
    flag = 1
    while flag:
        # do a level-set traversal from x
        level, order = _breadth_first_search(ind, ptr, x, level, order)
        # select node in last level with min degree
        maxlevel = max(level)
        lastnodes = np.where(level == maxlevel)[0]
        lastnodesvalence = valence[lastnodes]
        minlastnodesvalence = min(lastnodesvalence)
        node = np.where(lastnodesvalence == minlastnodesvalence)[0][0]
        node = lastnodes[node]

        # if d(x,y)>delta, set, and go to bfs above
        if level[node] > delta:
            x = node
            delta = level[node]
        else:
            flag = 0
    return x, order, level