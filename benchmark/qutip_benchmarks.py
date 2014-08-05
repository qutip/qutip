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

import platform
import json
import numpy as np
from scipy import *
from qutip import *
from tests import *

#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-profiler",
                    help="run profiler on qutip benchmarks",
                    action='store_true')
parser.add_argument("-o", "--output-file",
                    help="file name for benchmark output",
                    default="qutip-benchmarks.json", type=str)
parser.add_argument("-N", "--runs",
                    help="number of times to perform each benchmark",
                    default=3, type=int)
args = parser.parse_args()

qutip_info = [{'label': 'QuTiP', 'value': qutip.__version__},
              {'label': 'Python', 'value': platform.python_version()},
              {'label': 'NumPy', 'value': numpy.__version__},
              {'label': 'SciPy', 'value': scipy.__version__}]


#---------------------
# Run Python Benchmarks
#---------------------
def run_tests(N):
    # setup list for python times
    python_times = []
    test_names = []

    for test_function in test_function_list():
        try:
            out = test_function(N)
        except:
            out = [["unsupported"], [0.0]]
        test_names += out[0]
        python_times += out[1]

    return python_times, test_names

if args.run_profiler:
    import cProfile
    cProfile.run('run_tests(1)', 'qutip_benchmarks_profiler')
    import pstats
    p = pstats.Stats('qutip_benchmarks_profiler')
    p.sort_stats('cumulative').print_stats(50)

else:
    times, names = run_tests(args.runs)

    data = [{'name': names[n], 'time': times[n]} for n in range(len(names))]

    qutip_info.append({'label': 'Acc. time', 'value': "%.2f s" % sum(times)})

    qutip_bm = {"info": qutip_info, "data": data}

    with open(args.output_file, "w") as outfile:
        json.dump(qutip_bm, outfile, sort_keys=True, indent=4, ensure_ascii=True)
