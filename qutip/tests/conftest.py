import pytest
import functools
import os
import tempfile
import numpy as np


def _add_repeats_if_marked(metafunc):
    """
    If the metafunc is marked with the 'repeat' mark, then add the requisite
    number of repeats via parametrisation.
    """
    marker = metafunc.definition.get_closest_marker('repeat')
    if marker:
        count = marker.args[0]
        metafunc.fixturenames.append('_repeat_count')
        metafunc.parametrize('_repeat_count',
                             range(count),
                             ids=["rep({})".format(x+1) for x in range(count)])


def _skip_cython_tests_if_unavailable(item):
    """
    Skip the current test item if Cython is unavailable for import, or isn't a
    high enough version.
    """
    if item.get_closest_marker("requires_cython"):
        # importorskip rather than mark.skipif because this way we get pytest's
        # version-handling semantics.
        pytest.importorskip('Cython', minversion='0.14')
        pytest.importorskip('filelock')


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc):
    _add_repeats_if_marked(metafunc)


def pytest_runtest_setup(item):
    _skip_cython_tests_if_unavailable(item)


@pytest.fixture
def in_temporary_directory():
    """
    Creates a temporary directory for the lifetime of the fixture and changes
    into it.  All relative paths used will be in the temporary directory, and
    everything will automatically be cleaned up at the end of the fixture's
    life.
    """
    previous_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temporary_dir:
        os.chdir(temporary_dir)
        yield
        # pytest should catch exceptions occuring in functions using the
        # fixture, so this should always be called.  We want it here rather
        # than outside to prevent the case of the directory failing to be
        # removed because it is 'busy'.
        os.chdir(previous_dir)
