import pytest
import numpy as np
import uuid
import qutip
from pathlib import Path

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


@pytest.mark.parametrize('use_path', [True, False], ids=['Path', 'str'])
@pytest.mark.parametrize('suffix', ['', '.qu', '.dat'])
def test_qsave_qload(use_path, suffix):
    ops_in = [qutip.sigmax(),
              qutip.num(_dimension),
              qutip.coherent_dm(_dimension, 1j)]
    filename = _random_file_name() + suffix
    if use_path:
        filename = Path.cwd() / filename
    qutip.qsave(ops_in, filename)
    ops_out = qutip.qload(filename)
    assert ops_in == ops_out
    # check that the file was saved with the correct name:
    assert Path(str(filename) + ".qu").exists()
