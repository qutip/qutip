# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import pytest
import os
import tempfile


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


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc):
    _add_repeats_if_marked(metafunc)


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


@pytest.fixture
def tmpfile():
    with tempfile.NamedTemporaryFile() as file:
        yield file
