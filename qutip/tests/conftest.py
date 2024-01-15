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


def _patched_build_err_msg(arrays, err_msg, header='Items are not equal:',
                           verbose=True, names=('ACTUAL', 'DESIRED'),
                           precision=8):
    """
    Taken almost verbatim from `np.testing._private.utils`, except this version
    doesn't truncate output if it's longer than three lines.

    LICENCE
    -------
    Copyright (c) 2005-2020, NumPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the NumPy Developers nor the names of any
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    msg = ['\n' + header]
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79-len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, np.ndarray):
                # precision argument is only needed if the objects are ndarrays
                r_func = functools.partial(np.core.array_repr,
                                           precision=precision)
            else:
                r_func = repr

            try:
                # [diff] Remove truncation threshold from array output.
                with np.printoptions(threshold=np.inf):
                    r = r_func(a)
            except Exception as exc:
                r = '[repr failed for <{}>: {}]'.format(type(a).__name__, exc)
            # [diff] The original truncates the output to 3 lines here.
            msg.append(f' {names[i]}: {r}')
    return '\n'.join(msg)


# Find the private module used by numpy to store its testing utility functions
# so that we can monkeypatch the error messages to be more verbose.  QuTiP
# supports numpy from 1.12 upwards, so we have to search.
_numpy_private_utils_paths = [
    ['_private', 'utils'],    # 1.15.0 <= x
    ['nose_tools', 'utils'],  # 1.14.0 <= x < 1.15.0
    ['utils'],                # 1.14.0 > x
]
for possible_path in _numpy_private_utils_paths:
    try:
        module = np.testing
        for submodule in possible_path:
            module = getattr(module, submodule)
        _numpy_private_utils = module
        break
    except NameError:
        pass
else:
    # If we can't locate it for some reason, then we don't attempt to patch.
    _numpy_private_utils = None

if _numpy_private_utils is not None:
    @pytest.fixture(autouse=True)
    def do_not_truncate_numpy_output(monkeypatch):
        """
        Monkeypatch the internal numpy function used for printing arrays so
        that we get full output that isn't cut off after three lines.
        """
        with monkeypatch.context() as patch:
            patch.setattr(np.testing._private.utils, "build_err_msg",
                          _patched_build_err_msg)
            # Yield inside context manager just to minimize the amount of time
            # we've monkeypatched such a core library (and a private function!)
            yield


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
