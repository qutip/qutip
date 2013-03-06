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
if False:
    if sys.platform == 'darwin':
        sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
        sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
    else:
        sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
        sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)

# read matlab versions
matlab_version = csv.reader(open('matlab_version.csv'), dialect='excel')
for row in matlab_version:
    matlab_info = [{'label': 'Matlab', 'value': row[0]},
                   {'label': 'Type', 'value': row[1]}]

# read in matlab results
times = np.genfromtxt('matlab_benchmarks.csv', delimiter=',')
    
data = [{'name': "test%d" % n, 'time': times[n]} for n in range(len(times))]

matlab_bm = {"info": matlab_info, "data": data}

with open(args.output_file, "w") as outfile:
    json.dump(matlab_bm, outfile, sort_keys=True, indent=4, encoding='ascii')
