#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint
from qutip.core.data.base import idxint_dtype

cnp.import_array()

cdef cnp.ndarray[double] _max_row_weights(
        double[::1] data,
        int[::1] inds,
        int[::1] ptrs,
        int ncols):
    """
    Finds the largest abs value in each matrix column
    and the max. total number of elements in the cols (given by weights[-1]).

    Here we assume that the user already took the ABS value of the data.
    This keeps us from having to call abs over and over.
    """
    cdef cnp.ndarray[double] weights = np.zeros(ncols + 1, dtype=np.float64)
    cdef int ln, mx, ii, jj
    cdef double weight, current
    mx = 0
    for jj in range(ncols):
        ln = (ptrs[jj + 1] - ptrs[jj])
        if ln > mx:
            mx = ln
        weight = data[ptrs[jj]]
        for ii in range(ptrs[jj] + 1, ptrs[jj + 1]):
            current = data[ii]
            if current > weight:
                weight = current
        weights[jj] = weight
    weights[ncols] = mx
    return weights


def weighted_bipartite_matching(
        double[::1] data,
        int[::1] inds,
        int[::1] ptrs,
        int n):
    """
    Here we assume that the user already took the ABS value of the data.
    This keeps us from having to call abs over and over.
    """
    cdef cnp.ndarray[idxint] visited = np.zeros(n, dtype=idxint_dtype)
    cdef cnp.ndarray[idxint] queue = np.zeros(n, dtype=idxint_dtype)
    cdef cnp.ndarray[idxint] previous = np.zeros(n, dtype=idxint_dtype)
    cdef cnp.ndarray[idxint] match = -1 * np.ones(n, dtype=idxint_dtype)
    cdef cnp.ndarray[idxint] row_match = -1 * np.ones(n, dtype=idxint_dtype)
    cdef cnp.ndarray[double] weights = _max_row_weights(data, inds, ptrs, n)
    cdef cnp.ndarray[idxint] order = np.argsort(-weights[:n]).astype(idxint_dtype)
    cdef cnp.ndarray[idxint] row_order = np.zeros(int(weights[n]),
                                                  dtype=idxint_dtype)
    cdef cnp.ndarray[double] temp_weights = np.zeros(int(weights[n]),
                                                     dtype=np.float64)
    cdef int queue_ptr, queue_col, queue_size, next_num
    cdef int i, j, zz, ll, kk, row, col, temp, eptr, temp2

    next_num = 1
    for i in range(n):
        zz = order[i]  # cols with largest abs values first
        if (match[zz] == -1 and (ptrs[zz] != ptrs[zz + 1])):
            queue[0] = zz
            queue_ptr = 0
            queue_size = 1

            while (queue_size > queue_ptr):
                queue_col = queue[queue_ptr]
                queue_ptr += 1
                eptr = ptrs[queue_col + 1]

                # get row inds in current column
                temp = ptrs[queue_col]
                for kk in range(eptr - ptrs[queue_col]):
                    row_order[kk] = inds[temp]
                    temp_weights[kk] = data[temp]
                    temp += 1

                # linear sort by row weight
                for kk in range(1, (eptr - ptrs[queue_col])):
                    val = temp_weights[kk]
                    row_val = row_order[kk]
                    ll = kk - 1
                    while (ll >= 0) and (temp_weights[ll] > val):
                        temp_weights[ll + 1] = temp_weights[ll]
                        row_order[ll + 1] = row_order[ll]
                        ll -= 1
                    temp_weights[ll + 1] = val
                    row_order[ll + 1] = row_val

                # go through rows by decending weight
                temp2 = (eptr - ptrs[queue_col]) - 1
                for kk in range(eptr - ptrs[queue_col]):
                    row = row_order[temp2 - kk]
                    temp = visited[row]
                    if temp != next_num and temp != -1:
                        previous[row] = queue_col
                        visited[row] = next_num
                        col = row_match[row]
                        if col == -1:
                            while row != -1:
                                col = previous[row]
                                temp = match[col]
                                match[col] = row
                                row_match[row] = col
                                row = temp
                            next_num += 1
                            queue_size = 0
                            break
                        queue[queue_size] = col
                        queue_size += 1

            if match[zz] == -1:
                for j in range(1, queue_size):
                    visited[match[queue[j]]] = -1

    return match
