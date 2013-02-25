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

import scipy
import time
from numpy.testing import assert_, run_module_suite

from qutip import *


def _func(x):
    time.sleep(scipy.rand() * 0.25)  # random delay
    return x**2


def test_parfor1():
    "parfor"

    x = arange(10)
    y1 = list(map(_func, x))
    y2 = parfor(_func, x)

    assert_((array(y1) == array(y2)).all())


if __name__ == "__main__":
    run_module_suite()
