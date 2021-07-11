import os
import numpy as np
import qutip.settings as qset


def check_use_openmp(options):
    """
    Check to see if OPENMP should be used in dynamic solvers.
    """
    force_omp = False
    if qset.has_openmp and options.use_openmp is None:
        options.use_openmp = True
        force_omp = False
    elif qset.has_openmp and options.use_openmp == True:
        force_omp = True
    elif qset.has_openmp and options.use_openmp == False:
        force_omp = False
    elif qset.has_openmp == False and options.use_openmp == True:
        raise Exception('OPENMP not available.')
    else:
        options.use_openmp = False
        force_omp = False
    #Disable OPENMP in parallel mode unless explicitly set.    
    if not force_omp and os.environ['QUTIP_IN_PARALLEL'] == 'TRUE':
        options.use_openmp = False


def use_openmp():
    """
    Check for using openmp in general cases outside of dynamics
    """
    if qset.has_openmp and os.environ['QUTIP_IN_PARALLEL'] != 'TRUE':
        return True
    else:
        return False


def openmp_components(ptr_list):
    return np.array([ptr[-1] >= qset.openmp_thresh for ptr in ptr_list], dtype=bool)
