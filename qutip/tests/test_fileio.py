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
import numpy as np
import uuid
import qutip

# qsave _always_ appends a suffix to the file name at the time of writing, but
# in case this changes in the future, to ensure that we never leak a temporary
# file into the user's folders, we simply apply these tests in a temporary
# directory.  Windows also does not allow temporary files to be opened multiple
# times, so using a temporary directory is best.
pytestmark = [pytest.mark.usefixtures("in_temporary_directory")]

_dimension = 10

def _random_file_name():
    return "_" + str(uuid.uuid4())


class Test_file_data_store_file_data_read:
    # Tests parametrised seprately to give nicer descriptions in verbose mode.

    def case(self, filename, kwargs):
        data = 1 - 2*np.random.rand(_dimension, _dimension)
        if kwargs.get('numtype', 'complex') == 'complex':
            data = data * (0.5*0.5j)
        qutip.file_data_store(filename, data, **kwargs)
        out = qutip.file_data_read(filename)
        np.testing.assert_allclose(data, out, atol=1e-8)

    def test_defaults(self):
        return self.case(_random_file_name(), {})

    @pytest.mark.parametrize("type_", ["real", "complex"])
    @pytest.mark.parametrize("format_", ["decimal", "exp"])
    def test_type_format(self, type_, format_):

        kwargs = {'numtype': type_, 'numformat': format_}
        return self.case(_random_file_name(), kwargs)

    @pytest.mark.parametrize("separator", [",", ";", "\t", " ", " \t "],
                             ids=lambda x: "'" + x + "'")
    def test_separator_detection(self, separator):
        kwargs = {'numtype': 'complex', 'numformat': 'exp', 'sep': separator}
        return self.case(_random_file_name(), kwargs)


def test_qsave_qload():
    ops_in = [qutip.sigmax(),
              qutip.num(_dimension),
              qutip.coherent_dm(_dimension, 1j)]
    filename = _random_file_name()
    qutip.qsave(ops_in, filename)
    ops_out = qutip.qload(filename)
    assert ops_in == ops_out
