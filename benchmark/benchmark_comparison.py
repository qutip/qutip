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
parser.add_argument("-i", "--benchmark-input",
                    help="file name for benchmark input",
                    default="qutip-benchmark.json", type=str)
parser.add_argument("-r", "--benchmark-reference",
                    help="file name for benchmark refernce",
                    default="matlab-benchmark.json", type=str)
parser.add_argument("-o", "--output-file",
                    help="file name for benchmark comparison output",
                    default="benchmark_data.json", type=str)
args = parser.parse_args()

#
# get hardware info
#
platform = [hardware_info()]

#
# read in benchmark files
#
f = open(args.benchmark_input)
mb1_data = json.loads(f.read())
f.close()

f = open(args.benchmark_reference)
mb2_data = json.loads(f.read())
f.close()

#
# generate comparison data for the two benchmarks
#
data = []
for n in range(len(mb1_data["data"])):
    name = mb1_data["data"][n]["name"]
    dt1 = mb1_data["data"][n]["time"]
    dt2 = mb2_data["data"][n]["time"]
    factor = dt1 / dt2
    data.append({'name': str(name), 'factor': factor})   

f = open(args.output_file, "w")
f.write('data = ' + str(data) + ';\n')
f.write('platform = ' + str(platform) + ';\n')
f.write('bm1_info = ' + str(mb1_data["info"]) + ';\n')
f.write('bm2_info= ' + str(mb2_data["info"]) + ';\n')
f.close()
