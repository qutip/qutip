import os
from qutip.settings import settings as qset


def check_use_openmp(options):
    """
    Check to see if OPENMP should be used in dynamic solvers.
    """
    # TODO: sort this out.
    return False
    """
    force_omp = False
    if qset.has_openmp:
        if options.use_openmp is None:
            options.use_openmp = True
        else:
            force_omp = bool(options.use_openmp)
    elif (not qset.has_openmp) and options.use_openmp:
        raise Exception('OPENMP not available.')
    else:
        options.use_openmp = False
        force_omp = False
    # Disable OPENMP in parallel mode unless explicitly set.
    if not force_omp and os.environ['QUTIP_IN_PARALLEL'] == 'TRUE':
        options.use_openmp = False"""


def use_openmp():
    """
    Check for using openmp in general cases outside of dynamics
    """
    return False
    if qset.has_openmp and os.environ['QUTIP_IN_PARALLEL'] != 'TRUE':
        return True
    else:
        return False


def openmp_components(ptr_list):
    return np.array([False for ptr in ptr_list], dtype=bool)
    return np.array([ptr[-1] >= qset.openmp_thresh for ptr in ptr_list], dtype=bool)
