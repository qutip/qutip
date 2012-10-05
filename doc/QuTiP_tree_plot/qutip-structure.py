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
               "floquet":           0,
               "essolve":           0,
               "correlation":       0,
               "bloch_redfield":    0,
               "steady":            0,
               "rhs_generate":      0,
               "propagator":        0,
               "eseries":           0,
               "odeconfig":         1, # options and settings
               "settings":          1,
               "odechecks":         1,
               "Odedata":           1,
               "Odeoptions":        1,
               "Bloch":             2, # visualization
               "sphereplot":        2,
               "orbital":           2,
               "graph":             2,
               "wigner":            2,
               "tomography":        2,
               "operators":         3, # operators
               "superoperator":     3,
               "states":            4, # states
               "qstate":            4,
               "rand":              4,
               "three_level_atom":  4, 
               "gates":             5, # gates
               "entropy":           6, # measures
               "metrics":           6,
               "fileio":            8, # utilities
               "sparse":            8,
               "simdiag":           8,
               "demos":             8,
               "about":             8,
               "testing":           8,
               "parfor":            8,
               "clebsch":           8,
               "Qobj":              7, # core
               "expect":            7,
               "tensor":            7,
               "ptrace":            7,
               "istests":           7
               }


module_list  = []

for root, dirs, files in os.walk(qutip_root):  
    
    if not ".svn" in root and root == "../../qutip/qutip":

        for f in files:
            if f[-3:] == ".py" and f[0] != "_":

                module = f[:-3]
                
                idx   = module_cmap[module] if module in module_cmap else -1                 
                color = colors[idx] if idx >= 0 else "black"

                symbol_list = []

                cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (qutip_root, f)
                for name in os.popen(cmd).readlines():
                    if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                        symbol_list.append({"name": name.strip(), "size": 1000, "color": color})

                module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})


module_list_sorted = sorted(module_list, key=lambda x: x["idx"])
qutip_struct = {"name": "QuTiP", "children": module_list_sorted, "size": 2000}

print json.dumps(qutip_struct, sort_keys=True, indent=4)

