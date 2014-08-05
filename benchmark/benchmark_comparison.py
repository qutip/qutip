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
                    default="qutip-benchmarks.json", type=str)
parser.add_argument("-r", "--benchmark-reference",
                    help="file name for benchmark refernce",
                    default="matlab-benchmarks.json", type=str)
parser.add_argument("-o", "--output-file",
                    help="file name for benchmark comparison output",
                    default="benchmark_data.json", type=str)
args = parser.parse_args()

#
# get hardware info
#
platform = [{'label': label, 'value': value}
            for label, value in hardware_info().items()]

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
    if dt2 > 0.0 and dt1 > 0.0:
        factor = dt2 / dt1
        data.append({'name': str(name), 'factor': factor})

f = open(args.output_file, "w")
f.write('data = ' + str(data) + ';\n')
f.write('platform = ' + str(platform).replace("u'", "'") + ';\n')
f.write('bm1_info = ' + str(mb1_data["info"]).replace("u'", "'") + ';\n')
f.write('bm2_info= ' + str(mb2_data["info"]).replace("u'", "'") + ';\n')
f.close()
