__all__ = ['BaseProgressBar', 'TextProgressBar',
           'EnhancedTextProgressBar', 'TqdmProgressBar',
           'HTMLProgressBar', 'progress_bars']

import time
import datetime
import sys
from qutip import settings


class BaseProgressBar(object):
    """
    An abstract progress bar with some shared functionality.

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = TextProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update()
            compute_with_n(n)
        pbar.finished()

    """

    def __init__(self, iterations=0, chunk_size=10, **kwargs):
        self._start(iterations, chunk_size)

    def _start(self, iterations, chunk_size=10, **kwargs):
        self.N = float(iterations)
        self.n = 0
        self.p_chunk_size = chunk_size
        self.p_chunk = chunk_size
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self):
        pass

    def total_time(self):
        return self.t_done - self.t_start

    def time_elapsed(self):
        return "%6.2fs" % (time.time() - self.t_start)

    def time_remaining_est(self, p):
        if 100 >= p > 0.0:
            t_r_est = (time.time() - self.t_start) * (100.0 - p) / p
        else:
            t_r_est = 0

        dd = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % \
            (dd.day - 1, dd.hour, dd.minute, dd.second)

        return time_string

    def finished(self):
        self.t_done = time.time()


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10, **kwargs):
        super()._start(iterations, chunk_size)

    def update(self):
        self.n += 1
        n = self.n
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            print("%4.1f%%." % p +
                  " Run time: %s." % self.time_elapsed() +
                  " Est. time left: %s" % self.time_remaining_est(p))
            sys.stdout.flush()
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        print("Total run time: %s" % self.time_elapsed())


class EnhancedTextProgressBar(BaseProgressBar):
    """
    An enhanced text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10, **kwargs):
        super()._start(iterations, chunk_size)
        self.fill_char = '*'
        self.width = 25

    def update(self):
        self.n += 1
        n = self.n
        percent_done = int(round(n / self.N * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        prog_bar = ('[' + self.fill_char * num_hashes +
                    ' ' * (all_full - num_hashes) + ']')
        pct_place = (len(prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        prog_bar = (prog_bar[0:pct_place] +
                    (pct_string + prog_bar[pct_place + len(pct_string):]))
        prog_bar += ' Elapsed %s / Remaining %s' % (
            self.time_elapsed().strip(),
            self.time_remaining_est(percent_done))
        print('\r', prog_bar, end='')
        sys.stdout.flush()

    def finished(self):
        self.t_done = time.time()
        print("\r", "Total run time: %s" % self.time_elapsed())


class TqdmProgressBar(BaseProgressBar):
    """
    A progress bar using tqdm module
    """

    def __init__(self, iterations=0, chunk_size=10, **kwargs):
        from tqdm.auto import tqdm
        self.pbar = tqdm(total=iterations, **kwargs)
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self):
        self.pbar.update()

    def finished(self):
        self.pbar.close()
        self.t_done = time.time()


class HTMLProgressBar(BaseProgressBar):
    """
    A simple HTML progress bar for using in IPython notebooks. Based on
    IPython ProgressBar demo notebook:
    https://github.com/ipython/ipython/tree/master/examples/notebooks

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = HTMLProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update()
            compute_with_n(n)
    """

    def __init__(self, iterations=0, chunk_size=1.0, **kwargs):
        from IPython.display import HTML, Javascript, display
        import uuid

        self.display = display
        self.Javascript = Javascript
        self.divid = str(uuid.uuid4())
        self.textid = str(uuid.uuid4())
        self.pb = HTML(
            '<div style="border: 2px solid grey; width: 600px">\n  '
            f'<div id="{self.divid}" '
            'style="background-color: rgba(121,195,106,0.75); '
            'width:0%">&nbsp;</div>\n'
            '</div>\n'
            f'<p id="{self.textid}"></p>\n'
        )
        self.display(self.pb)
        super()._start(iterations, chunk_size)

    def update(self):
        self.n += 1
        n = self.n
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            lbl = ("Elapsed time: %s. " % self.time_elapsed() +
                   "Est. remaining time: %s." % self.time_remaining_est(p))
            js_code = f"""
                document.getElementById('{self.divid}').style.width = '{p}%';
                document.getElementById('{self.textid}').textContent = '{lbl}';
            """
            self.display(self.Javascript(js_code))
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        lbl = "Elapsed time: %s" % self.time_elapsed()
        js_code = f"""
            document.getElementById('{self.divid}').style.width = '{100.0}%';
            document.getElementById('{self.textid}').textContent = '{lbl}';
        """
        self.display(self.Javascript(js_code))


progress_bars = {
    "Enhanced": EnhancedTextProgressBar,
    "enhanced": EnhancedTextProgressBar,
    "Text": TextProgressBar,
    "text": TextProgressBar,
    True: TextProgressBar,
    "Tqdm": TqdmProgressBar,
    "tqdm": TqdmProgressBar,
    "Html": HTMLProgressBar,
    "html": HTMLProgressBar,
    "base": BaseProgressBar,
    "": BaseProgressBar,
    False: BaseProgressBar,
    None: BaseProgressBar,
}
