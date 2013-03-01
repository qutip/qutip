import sys
import csv
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time
from tests import *
#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qutip-only",
                    help="only run qutip benchmarks",
                    action='store_true')
parser.add_argument("--run-profiler",
                    help="run profiler on qutip benchmarks",
                    action='store_true')
args = parser.parse_args()

# setup hardware list
platform=[hardware_info()]

# setup matlab info lists
if not args.qutip_only:

    if sys.platform=='darwin':
        sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
    else:
        sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
    matlab_version = csv.reader(open('matlab_version.csv'),dialect='excel')
    for row in matlab_version:
        matlab_info=[{'version': row[0],'type': row[1]}]

qutip_info=[{'qutip':qutip.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__}]
#---------------------
#Run Python Benchmarks
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
    
    if not args.qutip_only:
        # Call matlab benchmarks (folder must be in Matlab path!!!)
        if sys.platform=='darwin':
            sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
            sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
        else:
            sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
            sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
    
    # read in matlab results
    matlab_times = genfromtxt('matlab_benchmarks.csv', delimiter=',')
    
    factors=matlab_times/array(python_times)
    data=[]
    for ii in range(len(test_names)):
        entry={'name':test_names[ii],'factor':factors[ii]}
        data.append(entry)

    f = open("benchmark_data.js", "w")
    f.write('data = ' + str(data) + ';\n')
    f.write('platform = ' + str(platform) + ';\n')
    f.write('qutip_info = ' + str(qutip_info) + ';\n')
    f.write('matlab_info= ' + str(matlab_info) + ';\n')
    f.close()
