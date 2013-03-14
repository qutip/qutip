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


def test_unit_conversions():
    "utilities: unit conversions"

    T = np.random.rand() * 100.0
    diff = convert_meV_to_mK(convert_GHz_to_meV(convert_mK_to_GHz(T))) - T
    assert_(abs(diff) < 1e-6)

    w = np.random.rand() * 100.0
    diff = convert_meV_to_GHz(convert_mK_to_meV(convert_GHz_to_mK(w))) - w
    assert_(abs(diff) < 1e-6)


if __name__ == "__main__":
    run_module_suite()
