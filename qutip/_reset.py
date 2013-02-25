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
This module resets the global properties in qutip.settings and the
odeconfig parameters.
"""


def _reset():
    import multiprocessing
    import os
    import qutip.settings
    from qutip.hardware_info import hardware_info
    qutip.settings.qutip_graphics = os.environ['QUTIP_GRAPHICS']
    qutip.settings.qutip_gui = os.environ['QUTIP_GUI']
    qutip.settings.auto_herm = True
    qutip.settings.auto_tidyup = True
    qutip.settings.auto_tidyup_atol = 1e-12
    qutip.settings.debug = False
    # set cpus using hardware_info
    info = hardware_info()
    if 'cpus' in info:
        qutip.settings.num_cpus = info['cpus']
    else:
        qutip.settings.num_cpus = multiprocessing.cpu_count()
