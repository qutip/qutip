import numpy as np
import pytest

from qutip.testing import get_test_generator
from qutip.tests.conftest import SEEDSEQ


@pytest.mark.parametrize("full_name", [True, False], ids=["nodeid", "name"])
@pytest.mark.parametrize("dtype", ["CSR", "Dense"])
def test_get_test_generator(random_generator, request, full_name, dtype):
    """
    ``get_test_generator`` recreates the ``random_generator`` fixture from the
    global seed and the test name, with or without the file name.
    """
    test_name = request.node.nodeid
    if not full_name:
        test_name = test_name.split("::")[-1]
    generator = get_test_generator(SEEDSEQ.entropy, test_name)
    np.testing.assert_array_equal(
        generator.random(10), random_generator.random(10)
    )
