import sys
import csv
import subprocess as sproc
from numpy import genfromtxt
from qutip import *
import numpy as np
from time import time
import qutip.settings as qset
from tests import *
#qset.auto_tidyup=False
#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qutip-only",
                    help="only run qutip benchmarks",
                    action='store_true')
args = parser.parse_args()

platform = [hardware_info()]

matlab_info=[{'version': row[0],'type': row[1]}]
qutip_info=[{'qutip':qutip.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__}]

f = open("qutip-benchmark.json")
mb1_data = json.loads(f.read())
f.close()

f = open("qutip-benchmark.json")
mb2_data = json.loads(f.read())
f.close()
    
factors = array(bm1_times) / array(bm2_times)

data = []
for ii in range(len(test_names)):
    entry = {'name': "test%d" % ii, 'factor': factors[ii]}
    data.append(entry)

#f = open("benchmark_data.js", "w")
print('data = ' + str(data) + ';\n')
print('platform = ' + str(platform) + ';\n')
print('bm1_info = ' + str(mb1_data["info"]) + ';\n')
print('bm2_info= ' + str(mb2_data["info"]) + ';\n')
#f.close()
