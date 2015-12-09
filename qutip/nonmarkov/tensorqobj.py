"""
This module contains

- A wrapper class around qutip's Qobj class to allow for some tensor
  manipulations. Mostly to be able to perform generalized partial trace.
- Misc. super-operators of the form I + \sqrt{\Delta t}A + \Delta t B that are
  used to build tensor-networks for computing expectation values for the
  output field

"""

import numpy as np
import scipy as sp

import qutip as qt


# Flag to decide if we are going to use cython for fast computation of
# tensor network generalized partial trace.
usecython = False
if usecython:
    import tnintegrate_c


class TensorQobj(qt.Qobj):
    """
    A wrapper around qutip Qobj to be able to view it as a tensor.
    This class is meant for representing super-operators.
    Each index of the tensor is a "double index" of dimension
    2*dim(H), i.e., the dimension of L(H). This convention
    is chosen to make it easier to work with qutip Qobj's of
    "super" type, representing superoperators. For example, this
    is consistent with how qutip's reshuffle works.
    """

    @property
    def nsys(self):
        # number of subsystems for the underlying Hilbert space
        return len(self.reshuffle().dims[0])

    @property
    def rank(self):
        # rank of tensor
        # each index is really a "double index"
        return self.nsys*2

    @property
    def sysdim(self):
        # dim of H
        return self.dims[0][0][0]

    @property
    def superdim(self):
        # dim of L(H)
        return self.sysdim**2

    def __mul__(self, other):
        return TensorQobj(super(TensorQobj,self).__mul__(other))

    def __rmul__(self, other):
        return TensorQobj(super(TensorQobj,self).__rmul__(other))

    def reshuffle(self):
        return TensorQobj(qt.reshuffle(self))

    def getmatrixindex(self,indices):
        # returns matrix indices of T given tensor indices
        # each subsystem has dimension self.superdim (double index)
        # indices = [i1,j1,...,iM,jM]
        if not len(indices) == self.rank:
            raise ValueError("number of indices do not match rank of tensor")
        ii = 0
        jj = 0
        idx = list(indices)
        for l in range(int(len(indices)/2)):
            j = idx.pop()
            i = idx.pop()
            ii += i*self.superdim**l
            jj += j*self.superdim**l
        return ii,jj

    def gettensorelement(self,indices):
        # return element given tensor indices
        return self.reshuffle()[self.getmatrixindex(indices)]

    def loop(self):
        # return T reduced by one subsystem by summing over 2 indices
        out = TensorQobj(dims=[[[self.sysdim]*(self.nsys-1)]*2]*2)
        idx = [0]*out.rank
        indices = []
        sumindices = []
        for cnt in range(out.superdim**(out.rank)):
            indices.append(list(idx))
            sumindices.append(list(idx[0:-1] + [0,0] + idx[-1:]))
            idx[0] += 1
            for i in range(len(idx)-1):
                if idx[i] > self.superdim-1:
                    idx[i] = 0
                    idx[i+1] += 1
        out2 = out.reshuffle()
        if usecython:
            indices = np.array(indices,dtype=np.int_)
            sumindices = np.array(sumindices,dtype=np.int_)
            indata = np.array(self.reshuffle().data.toarray(),dtype=np.complex_)
            data = tnintegrate_c.loop(indata,out2.shape,
                        indices,sumindices,self.superdim)
            out2 = TensorQobj(data,dims=out2.dims)
        else:
            for idx in indices:
                i,j = out.getmatrixindex(idx)
                for k in range(self.superdim):
                    sumidx = idx[0:-1] + [k,k] + idx[-1:]
                    out2.data[i,j] += self.gettensorelement(sumidx)
        #return out2
        return out2.reshuffle()
