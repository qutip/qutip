import qutip.settings as qset

if qset.has_mkl:
    from qutip.mkl.utilities import _set_mkl
    from qutip.mkl.spmv import mkl_spmv
    from qutip.mkl.spsolve import (mkl_splu, mkl_spsolve)
else:
    from qutip.mkl.utilities import _set_mkl