import qutip.settings as qset
from qutip.mkl.utilities import _set_mkl

_set_mkl()

if qset.mkl_lib is None:
    pass
else:
    from qutip.mkl.spmv import mkl_spmv
    from qutip.mkl.spsolve import mkl_spsolve