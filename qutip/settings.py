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
"""
This module contains settings for the QuTiP GUI,multiprocessing, and
tidyup functionality.

"""
# QuTiP Graphics (set at qutip import)
qutip_graphics = None
# QuTiP GUI selection (set at qutip import)
qutip_gui = None
# use auto tidyup
auto_tidyup = True
# detect hermiticity
auto_herm = True
# use auto tidyup absolute tolerance
auto_tidyup_atol = 1e-12
# number of cpus (set at qutip import)
num_cpus = 1
# flag indicating if fortran module is installed
fortran = False
# debug mode for development
debug = False


def reset():
    from qutip._reset import _reset
    _reset()


def load_rc_file(rc_file):
    """
    Load settings for the qutip RC file, by default .qutiprc in the user's home
    directory.
    """
    global qutip_graphics, qutip_gui, auto_tidyup, auto_herm, \
        auto_tidyup_atol, num_cpus, debug

    with open(rc_file) as f:
        for line in f.readlines():
            if line[0] != "#":
                var, val = line.strip().split("=")

                if var == "qutip_graphics":
                    qutip_graphics = "NO" if val == "NO" else "YES"

                elif var == "qutip_gui":
                    qutip_gui = val

                elif var == "auto_tidyup":
                    auto_tidyup = True if val == "True" else False

                elif var == "auto_tidyup_atol":
                    auto_tidyup_atol = float(val)

                elif var == "auto_herm":
                    auto_herm = True if val == "True" else False

                elif var == "num_cpus":
                    num_cpus = int(val)

                elif var == "debug":
                    debug = True if val == "True" else False
