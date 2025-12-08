from .about import about
from .settings import settings as qset

def run(full=False):
    """
    Run the test scripts for QuTiP.

    Parameters
    ----------
    full: bool
        If True run all test (30 min). Otherwise skip few variants of the
        slowest tests.
    """
    # Call about to get all version info printed with tests
    about()
    import pytest
    # real_num_cpu = qset.num_cpus
    # real_thresh = qset.openmp_thresh
    # if qset.has_openmp:
        # For travis which VMs have only 1 cpu.
        # Make sure the openmp version of the functions are tested.
    #     qset.num_cpus = 2
    #     qset.openmp_thresh = 100

    test_options = ["--verbosity=1", "--disable-pytest-warnings", "--pyargs"]
    if not full:
        test_options += ['-m', 'not slow']
    pytest.main(test_options + ["qutip"])
    # runs tests in qutip.tests module only

    # Restore previous settings
    # if qset.has_openmp:
    #     qset.num_cpus = real_num_cpu
    #     qset.openmp_thresh = real_thresh
