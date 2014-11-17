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

from __future__ import print_function
import time
import datetime
import sys
import threading

import qutip.settings

try:
    import tskmon
except ImportError:
    tskmon = None

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
        self.fill_char = '*'
        self.width = 25

    def update(self, n):
        percent_done = int(round(n / self.N * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        prog_bar = prog_bar[0:pct_place] + (pct_string + prog_bar[pct_place + len(pct_string):])
        prog_bar += ' Elapsed %s / Remaining %s' % (self.time_elapsed().strip(), self.time_remaining_est(percent_done))
        print ('\r', prog_bar, end='')
        sys.stdout.flush()

    def finished(self):
        self.t_done = time.time()
        print("\r","Total run time: %s" % self.time_elapsed())


class CompoundProgressBar(BaseProgressBar):

    def __init__(self, *bars):
        self._bars = bars

    def start(self, iterations, chunk_size=10):
        for bar in self._bars:
            bar.start(iterations, chunk_size)

    def update(self, n):
        for bar in self._bars:
            bar.update(n)

    def finished(self):
        for bar in self._bars:
            bar.finished()

class WebProgressBar(BaseProgressBar):
    def __init__(self, iterations=0, chunk_size=10, task_name="QuTiP Task"):
        super(WebProgressBar, self).__init__(iterations, chunk_size)
        self._client = tskmon.TskmonClient(qutip.settings.tskmon_token, app_name='QuTiP 3')
        self._wake_event = threading.Event()
        self._done = threading.Event()

        try:
            self._task = self._client.new_task(
                description=task_name,
                status=' ',
                max_progress=iterations, progress=0)
            self._thread = WebProgressThread(self._task, self._wake_event)
        except Exception as ex:
            print(ex)

    def start(self, iterations, chunk_size=10):
        try:
            self._task.update(progress=0, max_progress=iterations)
            self._thread.start()
        except Exception as ex:
            print(ex)

    def update(self, n):
        self._thread.dirty = True
        self._thread.progress = n
        self._wake_event.set()

    def finished(self):
        try:
            self._thread.done = True
            self._wake_event.set()
            self._task.delete()
        except Exception as ex:
            print(ex)

class WebProgressThread(threading.Thread):
    done = False
    dirty = False
    progress = 0

    def __init__(self, task, wake_event):
        super(WebProgressThread, self).__init__()
        self._task = task
        self._wake_event = wake_event

    def run(self):
        while True:
            if self.done:
                return
            if self.dirty:
                try:
                    self._task.update(progress=self.progress)
                    self.dirty = False
                    self._wake_event.clear()
                except Exception as ex:
                    print(ex)
            self._wake_event.wait()

