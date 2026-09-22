from qutip.about import about
from qutip.settings import settings as qset
import numpy as np


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


def get_test_generator(entropy, test_name):
    """
    Recreate the generator used in tests as random_generator fixture from the
    test global seed and test name.

    Parameters
    ----------
    entropy: int128
        Number printed at the start of the test suite:
        Run global seed: *********

    test_name: str
        Name of the test with parametrisation ids:
        "test_data_binary_operator[matmul-CSR-Dense]"
        The file name is not used.
    """
    if "::" in test_name:
        test_name = request.node.nodeid.split("::")[-1]
    test_name = test_name.encode("utf-8")
    name_hash = int(hashlib.sha256(test_name).hexdigest()[:32], 16)
    return np.random.default_rng( (entropy + name_hash) % 2**128 )
