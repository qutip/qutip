import sys
import csv
import json
import numpy as np
from qutip import *

#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qutip-only",
                    help="only run qutip benchmarks",
                    action='store_true')
args = parser.parse_args()

#
# get hardware info
#
platform = [hardware_info()]

#
# read in benchmark files
#
f = open("qutip-benchmark.json") # get from args
mb1_data = json.loads(f.read())
f.close()

f = open("qutip-benchmark.json") # get from args
mb2_data = json.loads(f.read())
f.close()

#
# generate comparison data for the two benchmarks
#
data = []
for n in range(len(mb1_data["data"])):
    name = mb1_data["data"][n]["name"]
    dt1 = mb1_data["data"][n]["factor"]
    dt2 = mb2_data["data"][n]["factor"]
    factor = dt1 / dt2
    data.append({'name': str(name), 'factor': factor})   

#f = open("benchmark_data.js", "w") # get filename from args
print('data = ' + str(data) + ';\n')
print('platform = ' + str(platform) + ';\n')
print('bm1_info = ' + str(mb1_data["info"]) + ';\n')
print('bm2_info= ' + str(mb2_data["info"]) + ';\n')
#f.close()
