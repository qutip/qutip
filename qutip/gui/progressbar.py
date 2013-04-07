# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import time

class TextProgressBar():
    """
    A simple text-based progress bar.

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = TextProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update(n)
            compute_with_n(n)
        pbar.finished()
    """

    def __init__(self, iterations=0, chunk_size=10):
        self.start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10):
        self.N = float(iterations)
        self.t_start = time.time()
        self.p_chunk_size = chunk_size
        self.p_chunk = 0

    def update(self, n):
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            print("%.2f" % p)
            self.p_chunk += self.p_chunk_size

    def finished(self):
       self.t_done = time.time()
       print("Elapsed time = %.2fs" % (self.t_done - self.t_start))
