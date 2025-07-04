import pytest
pytest.importorskip("IPython")

from qutip.ipynbtools import version_table

@pytest.mark.parametrize('verbose', [False, True])
def test_version_table(verbose):
    html_data = version_table(verbose=verbose).data
    assert "<th>Software</th>" in html_data
    assert "<th>Version</th>" in html_data
    assert "<td>QuTiP</td>" in html_data
    assert "<td>Numpy</td>" in html_data
    assert "<td>SciPy</td>" in html_data
    assert "<td>matplotlib</td>" in html_data
    assert "<td>IPython</td>" in html_data
    if verbose:
        assert "<td>Installation path</td>" in html_data
        if pytest.importorskip("getpass") is not None:
            assert "<td>User</td>" in html_data


@pytest.mark.skipif(not pytest.importorskip("Cython"), reason="cython not installed")
def test_version_table_with_cython():
    html_data = version_table().data
    assert "<td>Cython</td>" in html_data
