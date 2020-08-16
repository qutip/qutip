# First-class type imports

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
from .tidyup import *
from .trace import *
# For operations with mulitple related versions, we just import the module.
from . import norm, permute


# Set up the data conversions that are known by us.  All types covered by
# conversions will be made available for use in the dispatcher functions.

from .convert import to, create
to.add_conversions([
    (Dense, CSR, dense.from_csr, 1),
    (CSR, Dense, csr.from_dense, 1),
])

from .dispatch import Dispatcher


for dispatcher in to.dispatchers:
    dispatcher.rebuild_lookup()
