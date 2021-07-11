from __future__ import print_function

__all__ = ['BaseProgressBar', 'TextProgressBar', 'EnhancedTextProgressBar']

import time
import datetime
import sys


class BaseProgressBar(object):
    """
    An abstract progress bar with some shared functionality.

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = TextProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update(n)
            compute_with_n(n)
        pbar.finished()

    """

    def __init__(self, iterations=0, chunk_size=10):
        pass

    def start(self, iterations, chunk_size=10):
        self.N = float(iterations)
        self.p_chunk_size = chunk_size
        self.p_chunk = chunk_size
        self.t_start = time.time()

    def update(self, n):
        pass

    def time_elapsed(self):
        return "%6.2fs" % (time.time() - self.t_start)

    def time_remaining_est(self, p):
        if p > 0.0:
            t_r_est = (time.time() - self.t_start) * (100.0 - p) / p
        else:
            t_r_est = 0

        dd = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % \
            (dd.day - 1, dd.hour, dd.minute, dd.second)

        return time_string

    def finished(self):
        pass


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10):
        super(TextProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10):
        super(TextProgressBar, self).start(iterations, chunk_size)

    def update(self, n):
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

    def __init__(self, iterations=0, chunk_size=10):
        super(EnhancedTextProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10):
        super(EnhancedTextProgressBar, self).start(iterations, chunk_size)
        self.fill_char = '*'
        self.width = 25

    def update(self, n):
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
