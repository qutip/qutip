import sys
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

#qset.auto_tidyup=False
#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--qutip-only",
#                    help="only run qutip benchmarks",
#                    action='store_true')
parser.add_argument("--run-profiler",
                    help="run profiler on qutip benchmarks",
                    action='store_true')
args = parser.parse_args()

qutip_info = [{'qutip': qutip.__version__,
               'numpy': numpy.__version__,
               'scipy': scipy.__version__}]

#---------------------
# Run Python Benchmarks
#---------------------
def run_tests():
    #setup list for python times
    python_times=[]
    test_names=[]
    
    out=test_1();test_names+=out[0];python_times+=out[1]
    out=test_2();test_names+=out[0];python_times+=out[1]
    out=test_3();test_names+=out[0];python_times+=out[1]
    out=test_4();test_names+=out[0];python_times+=out[1]
    out=test_5();test_names+=out[0];python_times+=out[1]
    out=test_6();test_names+=out[0];python_times+=out[1]
    out=test_7();test_names+=out[0];python_times+=out[1]
    out=test_8();test_names+=out[0];python_times+=out[1]
    out=test_9();test_names+=out[0];python_times+=out[1]
    out=test_10();test_names+=out[0];python_times+=out[1]
    out=test_11();test_names+=out[0];python_times+=out[1]
    out=test_12();test_names+=out[0];python_times+=out[1]
    out=test_13();test_names+=out[0];python_times+=out[1]
    out=test_14();test_names+=out[0];python_times+=out[1]
    out=test_15();test_names+=out[0];python_times+=out[1]
    out=test_16();test_names+=out[0];python_times+=out[1]
    out=test_17();test_names+=out[0];python_times+=out[1]
    out=test_18();test_names+=out[0];python_times+=out[1]
    out=test_19();test_names+=out[0];python_times+=out[1]
    #return all results
    return python_times, test_names

if args.run_profiler:
    import cProfile
    cProfile.run('run_tests()', 'qutip_benchmarks_profiler')
    import pstats
    p = pstats.Stats('qutip_benchmarks_profiler')
    p.sort_stats('cumulative').print_stats(30)

else:
    python_times, test_names = run_tests()
        
    data = [{'name': test_names[n], 'factor': python_times[n]}
            for n in range(len(test_names))]
 
    qutip_bm = {"info": qutip_info, "data": data}

    print json.dumps(qutip_bm, sort_keys=True, indent=4)

