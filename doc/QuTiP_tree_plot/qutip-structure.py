#!/usr/bin/env python

import os
import sys
import shutil,fnmatch
import re
import subprocess
import warnings
from glob import glob
from os.path import splitext, basename, join as pjoin
from os import walk
import numpy as np

import json
import re

exclude_dirs = ["examples", "tests"]

# a list of exclude patterns for decrapated or poorly named
# internal functions
exclude_patterns = [r'.*_td', r'.*_es', r'.*_mc', r'.*_ode', r'_.*']

qutip_root = "../../qutip/qutip"

module_list  = []

for root, dirs, files in os.walk(qutip_root):  
    
    if not ".svn" in root and root == "../../qutip/qutip":

        for f in files:
            if f[-3:] == ".py" and f[0] != "_":

                module = f[:-3]

                symbol_list = []

                cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (qutip_root, f)
                for name in os.popen(cmd).readlines():
                    if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                        symbol_list.append({"name": name.strip(), "size": 2000})

                module_list.append({"name": module, "children": symbol_list})


qutip_struct = {"name": "QuTiP", "children": module_list}

print json.dumps(qutip_struct, sort_keys=True, indent=4)

