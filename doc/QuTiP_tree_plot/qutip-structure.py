#!/usr/bin/env python

import os
import json
import re

exclude_dirs = ["examples", "tests"]

# a list of exclude patterns for decrapated or poorly named
# internal functions
exclude_patterns = [r'.*_td', r'.*_es', r'.*_mc', r'.*_ode', r'_.*']

qutip_root = "../../qutip/qutip"

colors = ["#0B5FA5", "#043C6B", "#3F8FD2", # blue colors
          "#00AE68", "#007143", "#36D695", # green colors
          "#FF4500", "#692102", "#BF5730"
          ]

#          "#FF9400", "#A66000", "#FFAE40"
#          "#FF6F00", "#A64800", "#BF6E30"

module_cmap = {"mesolve":           0, # time evolution
               "mcsolve":           0,
               "sesolve":           0,
               "solver":            0,
               "stochastic":        0,
               "bloch_redfield":    0,
               "memorycascade":     0,
               "transfertensor":    0,
               "floquet":           0,
               "essolve":           0,
               "correlation":       0,
               "steadystate":       0,
               "rhs_generate":      0,
               "propagator":        0,
               "eseries":           0,
               "hsolve":            0,
               "rcsolve":           0,
               "heom":              0,               
               "scattering":        0,
               "piqs":              0,               
               "odeconfig":         1, # options and settings
               "settings":          1,
               "odechecks":         1,
               "odedata":           1,
               "odeoptions":        1,
               "bloch":             2, # visualization
               "bloch3d":           2,
               "sphereplot":        2,
               "orbital":           2,
               "visualization":     2,
               "wigner":            2,
               "distributions":     2,
               "tomography":        2,
               "operators":         3, # operators
               "superoperator":     3,
               "superop_reps":      3,
               "subsystem_apply":   3,
               "states":            4, # states
               "continuous_variables": 4,
               "qstate":            4,
               "random_objects":    4,
               "three_level_atom":  4, 
               "gates":             5, # gates
               "entropy":           6, # measures
               "metrics":           6,
               "countstat":         6,
               "fileio":            8, # utilities
               "utilities":         8,
               "ipynbtools":        8,
               "sparse":            8,
               "graph":             8,
               "simdiag":           8,
               "permute":           8,
               "demos":             8,
               "about":             8,
               "parallel":          8,
               "version":           8,
               "testing":           8,
               "parfor":            8,
               "hardware_info":     8,
               "qobj":              7, # core
               "expect":            7,
               "tensor":            7,
               "partial_transpose": 7,
               "ptrace":            7,
               "istests":           7,
               }


hidden_modules = ['testing','dimensions','logging_utils','matplotlib_utilities']

module_list  = []

num_items = 0

for root, dirs, files in os.walk(qutip_root):  
    
    if not ".svn" in root and root == "../../qutip/qutip":
        for f in files:
            if f[-3:] == ".py" and f[0] != "_" and f != "setup.py":

                module = f[:-3]
                if module not in hidden_modules:
                    idx   = module_cmap[module] if module in module_cmap else -1                 
                    color = colors[idx] if idx >= 0 else "black"

                    symbol_list = []

                    cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (qutip_root, f)
                    for name in os.popen(cmd).readlines():
                        if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                            symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
                            num_items+=1
                    module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})
        for d in dirs:
            if d in ['nonmarkov']:
                for root, dr, files in os.walk(qutip_root+'/'+d):  
                    for f in files:
                        if f[-3:] == ".py" and f[0] != "_" and f != "setup.py":
                            module = f[:-3]
                            idx   = module_cmap[module] if module in module_cmap else -1                 
                            color = colors[idx] if idx >= 0 else "black"

                            symbol_list = []

                            cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (qutip_root+'/'+d, f)
                            for name in os.popen(cmd).readlines():
                                if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                                    symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
                                    num_items+=1
                            module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})

module_list_sorted = sorted(module_list, key=lambda x: x["idx"])
qutip_struct = {"name": "QuTiP", "children": module_list_sorted, "size": 2000}

with open('d3_data/qutip.json', 'w') as outfile:
    json.dump(qutip_struct, outfile, sort_keys=True, indent=4)

print(num_items)