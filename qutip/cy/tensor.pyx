from cpython cimport bool

cdef struct pair: #if we every try to optimize the tensor contract, replace the index,value tuple pairs
    int index     #with this struct. should be good for a slight speed boost
    double complex value 

cdef _merge(list lol, bool ascend = True): #think of this as the merge step in a generalized merge-sort
    cdef int pivot
    cdef list res = []
    cdef list A = []
    cdef list B = []
    if len(lol) == 1: #lol = list of lists
        if not ascend: #this is a trick take advantage of the fact that poping from a list is faster
            lol[0].reverse()#than other list operations. Also I can avoid making redundant data copies
        return lol[0]
    else:
        pivot = len(lol)//2
        A = _merge(lol[:pivot],not ascend)
        B = _merge(lol[pivot:],not ascend)
        res = []
        while len(A)*len(B) > 0:
            if (A[-1] < B[-1] and ascend) or (A[-1] > B[-1] and not ascend):
                res.append(A.pop())
            else:
                res.append(B.pop())
        while len(A) > 0:
            res.append(A.pop())
        while len(B) > 0:
            res.append(B.pop())
        return res

def _tensor_contract_mainloop(data,tuple pairs,list adj_dims,tuple t_dims,list allidx,int new_h,int new_w):
    cdef int row = 0
    cdef int col
    cdef int idx
    cdef int newidx
    cdef int k
    cdef list lol = [[(-1,0)]] #a list of lists of (flat index, value) pairs. Each sub list is sorted by index
    cdef int w
    cdef int h
    h,w = data.get_shape()
    #second we check that the indices passed are valid
    for p in pairs: 
        if t_dims[p[0]] != t_dims[p[1]]:
            raise ValueError("Cannot contract over indices of different length.")
    oldp = allidx[0]
    for p in allidx[1:]:
        if p == oldp:#check that pairs does not contain overlapping indicies like [(i,j),(k,j)]. 
            raise ValueError("Cannot contract over overlapping pairs of indices (eg [(i,j),(k,j)]) or invalid pair (eg [(i,i)])")
        oldp = p
        
    #third we will loop through the sparse matrix data itself, mapping and adding data to a temporary data structure
    for dat_idx in range(len(data.data)):#this is efficent because incoming indices are already somewhat sorted.
        col = data.indices[dat_idx] #deduce the column value
        while dat_idx + 1 > data.indptr[row+1]: #and row value
            row += 1
        idx = row*w + col #totally flattened index

        
        accept = True #for every pair of indices we test that index i = index j 
        k=0           #since the result is usually false. early exit beats parallel testing.
        while accept and k < len(pairs): #used a while loop to allow early exit if false
            accept = (idx//adj_dims[pairs[k][0]])%t_dims[pairs[k][0]] == (idx//adj_dims[pairs[k][1]])%t_dims[pairs[k][1]]
            k+=1 #this test was derived from the numpy reshape documentation and painstakingly keeping track of indices

            
        if accept: #if all the indices matched then we add this element to the contracted tensor
            #but first we need to map the row and column to the contracted row and column
            #this is done by mapping the flat index to the contracted flat index
            newidx = idx
            for k in allidx:
                newidx = (adj_dims[k]*(newidx//(t_dims[k]*adj_dims[k]))) + (newidx % adj_dims[k]) #reassign the flat idx
                #this mapping was derived from the numpy reshape documentation and painstakingly keeping track of indices

            if newidx < lol[-1][-1][0]: #new index does not follow previous index
                lol += [[(newidx, data.data[dat_idx])]] #put it in a new sublist
            else: #new index does happen to follow previous index
                lol[-1] += [(newidx, data.data[dat_idx])] #put it at the end of the last sublist    
    lol = _merge(lol)
    
    #fourth we convert the temporary data structure back to CSR
    cdef list A = []
    cdef list IA = [0]
    cdef list JA = []
    cdef int prev_idx = -1
    for idx_val in lol[1:]:
        #fill in the CSR data
        if idx_val[0] == prev_idx:
            A[-1] += idx_val[1]
            if A[-1] == 0: #don't include 0 terms in sparse matrix. duh
                A.pop()
                JA.pop()
                IA[-1] += -1
                prev_idx = -1
        else:
            A.append(idx_val[1])
            JA.append(idx_val[0]%new_w)
            IA.extend([IA[-1]]*((idx_val[0]//new_w) - len(IA) + 2))
            IA[-1] += 1
            prev_idx = idx_val[0]
    IA.extend([IA[-1]]*(new_h+1-len(IA))) #fill remaining rows so dims match
    return (A,JA,IA)
