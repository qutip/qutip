import os
from qutip import *
os.environ['QUTIP_GRAPHICS']="NO"
path = os.path.dirname(__file__)

py_files = [f for f in os.listdir(path) if f.endswith('.py')]
py_files.remove('run_scripts.py')
py_files.remove('examples-angular.py') #figures have no cmd save feature
py_files.remove('examples-jc-model-wigner.py') #requires setting matplotlib Agg

for script in py_files:
    print 'Running '+script
    execfile(script)


print 'All scripts finished...'