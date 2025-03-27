from qutip.ui.progressbar import progress_bars
import pytest
import time


bars = ["base", "text", "Enhanced"]

try:
    import tqdm
    bars.append("tqdm")
except ImportError:
    bars.append(
        pytest.param("tqdm", marks=pytest.mark.skip("module not installed"))
    )

try:
    import IPython
    bars.append("html")
except ImportError:
    bars.append(
        pytest.param("html", marks=pytest.mark.skip("module not installed"))
    )


@pytest.mark.parametrize("pbar", bars)
def test_progressbar(pbar):
    N = 5
    bar = progress_bars[pbar](N)
    assert bar.total_time() < 0
    for _ in range(N):
        time.sleep(0.25)
        bar.update()
    bar.finished()
    assert bar.total_time() > 0


@pytest.mark.parametrize("pbar", bars)
def test_progressbar_too_few_update(pbar):
    N = 5
    bar = progress_bars[pbar](N)
    assert bar.total_time() < 0
    for _ in range(N-2):
        time.sleep(0.01)
        bar.update()
    bar.finished()
    assert bar.total_time() > 0


@pytest.mark.parametrize("pbar", bars)
def test_progressbar_too_many_update(pbar):
    N = 5
    bar = progress_bars[pbar](N)
    assert bar.total_time() < 0
    for _ in range(N+2):
        time.sleep(0.01)
        bar.update()
    bar.finished()
    assert bar.total_time() > 0


@pytest.mark.parametrize("pbar", bars[1:])
def test_progressbar_has_print(pbar, capsys):
    N = 2
    bar = progress_bars[pbar](N)
    bar.update()
    bar.finished()
    out, err = capsys.readouterr()
    assert out + err != ""
