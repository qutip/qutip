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
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import scipy
import time
from numpy.testing import assert_, run_module_suite

from qutip import *


def test_unit_conversions():
    "utilities: energy unit conversions"

    T = np.random.rand() * 100.0

    diff = convert_unit(convert_unit(T, orig="mK", to="GHz"),
                        orig="GHz", to="mK") - T
    assert_(abs(diff) < 1e-6)
    diff = convert_unit(convert_unit(T, orig="mK", to="meV"),
                        orig="meV", to="mK") - T
    assert_(abs(diff) < 1e-6)

    diff = convert_unit(convert_unit(convert_unit(T, orig="mK", to="GHz"),
                                     orig="GHz", to="meV"),
                        orig="meV", to="mK") - T
    assert_(abs(diff) < 1e-6)

    w = np.random.rand() * 100.0

    diff = convert_unit(convert_unit(w, orig="GHz", to="meV"),
                        orig="meV", to="GHz") - w
    assert_(abs(diff) < 1e-6)

    diff = convert_unit(convert_unit(w, orig="GHz", to="mK"),
                        orig="mK", to="GHz") - w
    assert_(abs(diff) < 1e-6)

    diff = convert_unit(convert_unit(convert_unit(w, orig="GHz", to="mK"),
                                     orig="mK", to="meV"),
                        orig="meV", to="GHz") - w
    assert_(abs(diff) < 1e-6)


if __name__ == "__main__":
    run_module_suite()
