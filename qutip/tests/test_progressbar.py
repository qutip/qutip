from qutip.ui.progressbar import progress_bars
import pytest
import time


# We offer multiple alias for each progressbar.
names = []
bars = []
for alias, bar in progress_bars.items():
    if bar not in bars:
        names.append(alias)
        bars.append(bar)


@pytest.mark.parametrize("pbar", names)
def test_progressbar(pbar, capsys):
    N = 5

    try:
        bar = progress_bars[pbar](N)
    except ImportError:
        # ipython or tqdm could be missing
        pytest.skip(reason="module not available")
    assert bar.total_time() < 0
    for _ in range(N):
        time.sleep(0.25)
        bar.update()
    bar.finished()
    assert bar.total_time() > 0
    out, err = capsys.readouterr()
    if pbar.lower() not in ["base"]:
        # Has an non error output
        # tqdm use error out
        assert out != "" or err != ""
