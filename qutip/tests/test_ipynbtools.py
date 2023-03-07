from qutip.ipynbtools import version_table
import sys


def test_version_table():
    if 'Cython' in sys.modules:
        assert 'Cython' in version_table().data
    else:
        assert 'Cython' not in version_table().data
