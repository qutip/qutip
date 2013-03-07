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

import sys
import platform
import csv
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time
import qutip.settings as qset
from tests import *
from scipy import *

import json

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
                    default=1, type=int)
args = parser.parse_args()

qutip_info = [{'label': 'QuTiP', 'value': qutip.__version__},
              {'label': 'Python', 'value': platform.python_version()},
              {'label': 'NumPy', 'value': numpy.__version__},
              {'label': 'SciPy', 'value': scipy.__version__}]

#---------------------
# Run Python Benchmarks
#---------------------
def run_tests(N):
    #setup list for python times
    python_times=[]
    test_names=[]
    
    out = test_1(N); test_names += out[0]; python_times += out[1]
    out = test_2(N); test_names += out[0]; python_times += out[1]
    out = test_3(N); test_names += out[0]; python_times += out[1]
    out = test_4(N); test_names += out[0]; python_times += out[1]
    out = test_5(N); test_names += out[0]; python_times += out[1]
    out = test_6(N); test_names += out[0]; python_times += out[1]
    out = test_7(N); test_names += out[0]; python_times += out[1]
    out = test_8(N); test_names += out[0]; python_times += out[1]
    out = test_9(N); test_names += out[0]; python_times += out[1]
    out = test_10(N); test_names += out[0]; python_times += out[1]
    out = test_11(N); test_names += out[0]; python_times += out[1]
    out = test_12(N); test_names += out[0]; python_times += out[1]
    out = test_13(N); test_names += out[0]; python_times += out[1]
    out = test_14(N); test_names += out[0]; python_times += out[1]
    out = test_15(N); test_names += out[0]; python_times += out[1]
    out = test_16(N); test_names += out[0]; python_times += out[1]
    out = test_17(N); test_names += out[0]; python_times += out[1]
    out = test_18(N); test_names += out[0]; python_times += out[1]
    out = test_19(N); test_names += out[0]; python_times += out[1]
    #return all results
    return python_times, test_names

if args.run_profiler:
    import cProfile
    cProfile.run('run_tests()', 'qutip_benchmarks_profiler')
    import pstats
    p = pstats.Stats('qutip_benchmarks_profiler')
    p.sort_stats('cumulative').print_stats(30)

else:
    times, names = run_tests(args.runs)
        
    data = [{'name': names[n], 'time': times[n]} for n in range(len(names))]
 
    qutip_bm = {"info": qutip_info, "data": data}

    with open(args.output_file, "w") as outfile:
        json.dump(qutip_bm, outfile, sort_keys=True, indent=4)
