import os
from qutip import *
path = os.path.dirname(__file__)

py_files = [f for f in os.listdir(path) if f.endswith('.py')]
py_files.remove('run_scripts.py')
        
print py_files

for script in py_files:
    execfile(script)


print 'All scripts finished...'