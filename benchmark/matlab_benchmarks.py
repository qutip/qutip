import sys
import csv
import subprocess as sproc
from numpy import genfromtxt
import numpy as np
from time import time
from tests import *

# Call matlab benchmarks (folder must be in Matlab path!!!)
if sys.platform=='darwin':
    sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
    sproc.call("/Applications/MATLAB_R2012b.app/bin/matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)
else:
    sproc.call("matlab -nodesktop -nosplash -r 'matlab_version; quit'",shell=True)
    sproc.call("matlab -nodesktop -nosplash -r 'matlab_benchmarks; quit'",shell=True)

# read matlab versions
matlab_version = csv.reader(open('matlab_version.csv'), dialect='excel')
for row in matlab_version:
    matlab_info=[{'version': row[0],'type': row[1]}]
    
# read in matlab results
matlab_times = genfromtxt('matlab_benchmarks.csv', delimiter=',')
    
data = [{'name': test_names[n], 'factor': matlab_times[n]}
        for n in range(len(test_names))]

matlab_mb = {"info": matlab_info, "data": data}

print json.dumps(matlab_bm, sort_keys=True, indent=4)
