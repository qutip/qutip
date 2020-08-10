from . import dense, csr

from .dense import Dense
from .csr import CSR
from .base import Data

from .add import *
from .adjoint import *
from .eigen import *
from .expect import *
from .expm import *
from .inner import *
from .kron import *
from .matmul import *
from .mul import *
from .pow import *
from .project import *
from .properties import *
from .ptrace import *
from .reshape import *
from .sub import *
from .tidyup import *
from .trace import *

# For operations with mulitple versions, we just import the module.
from . import norm, permute


# This is completely temporary - it will actually get replaced by a proper
# dispatcher in its own module, but for now I'm just trying to get CSR working
# within Qobj, and this is the fastest way to stub out this creation.

def create(arg, shape=None):
    import numpy as np
    import scipy.sparse

    if isinstance(arg, CSR):
        return arg.copy()
    if scipy.sparse.issparse(arg):
        return CSR(arg.tocsr(), shape=shape)
    # Promote 1D lists and arguments to kets, not bras by default.
    arr = np.array(arg, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise TypeError("input has incorrect dimensions: " + str(arr.shape))
    return csr.from_dense(dense.fast_from_numpy(arr))
