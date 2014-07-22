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
import subprocess as sproc
import json

import numpy as np

#
# command-line parsing
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output-file",
                    help="file name for benchmark output",
                    default="matlab-benchmarks.json", type=str)
args = parser.parse_args()

# Call matlab benchmarks (folder must be in Matlab path!!!)
if sys.platform == 'darwin':
    sproc.call("/Applications/MATLAB_R2014b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'", shell=True)
    sproc.call("/Applications/MATLAB_R2014b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'", shell=True)
else:
    sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'", shell=True)
    sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'", shell=True)

# read matlab versions
matlab_version = csv.reader(open('matlab_version.csv'), dialect='excel')
for row in matlab_version:
    matlab_info = [{'label': 'Matlab', 'value': row[0]},
                   {'label': 'Type', 'value': row[1]}]

# read in matlab results
times = np.genfromtxt('matlab_benchmarks.csv', delimiter=',')

matlab_info.append({'label': 'Acc. time', 'value': "%.2f s" % sum(times)})

data = [{'name': "test%d" % n, 'time': times[n]} for n in range(len(times))]

matlab_bm = {"info": matlab_info, "data": data}

with open(args.output_file, "w") as outfile:
    json.dump(matlab_bm, outfile, sort_keys=True, indent=4, ensure_ascii=True)
