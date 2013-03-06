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
platform = [{'label': label, 'value': value}
            for label, value in hardware_info().iteritems()]


#
# read in benchmark files
#
with open(args.benchmark_input) as f:
    mb1_data = json.load(f)

with open(args.benchmark_reference) as f:
    mb2_data = json.load(f)

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
f.write('platform = ' + str(platform).replace("u'", "'") + ';\n')
f.write('bm1_info = ' + str(mb1_data["info"]).replace("u'", "'") + ';\n')
f.write('bm2_info= ' + str(mb2_data["info"]).replace("u'", "'") + ';\n')
f.close()
