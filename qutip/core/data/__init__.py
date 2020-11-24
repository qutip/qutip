# First-class type imports

from . import dense, csr, csc
from .dense import Dense
from .csr import CSR
from .csc import CSC
from .base import Data

from .add import *
from .adjoint import *
from .constant import *
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
from .tidyup import *
from .trace import *
# For operations with mulitple related versions, we just import the module.
from . import norm, permute


# Set up the data conversions that are known by us.  All types covered by
# conversions will be made available for use in the dispatcher functions.

from .convert import to, create
to.add_conversions([
    (Dense, CSR, dense.from_csr, 1),
    (CSR, Dense, csr.from_dense, 1.4),
    (CSC, Dense, csc.from_dense, 1.4),
    (CSC, CSR, csc.from_csr, 1.0),
    (CSR, CSC, csc.to_csr, 1.0),
    (Dense, CSC, csc.to_dense, 1.0),
])

from .dispatch import Dispatcher
