#!/usr/bin/env python

import inspect
import pathlib
import warnings
import sys

import qutip

# This script currently relies on all packages being imported by the
#   import qutip
# command.  If in the future some packages are not imported, then you'll need
# to add more import lines below it to make sure they're all in.  We do this
# rather than file-based discovery so we have more access to information
# included by the import system, such as which names are meant to be public.
# It also means that we can import Cythonised modules to investigate their
# internals as well.

root_directory = pathlib.Path(qutip.__file__).parent

# This list needs to populated manually at the moment.  Each element of the
# list is a two-tuple (colour, modules), where the `colour` is the text colour
# in the output, and `modules` is a set of module names that will be that
# colour.  You can also put package names into the set of modules---any
# submodules of that package will inherit the same colour.  You don't need to
# include the "qutip." prefix to the modules.  It's a list not a dictionary
# because the order is important to the output.
module_groups = [
    # Solvers
    ("#0b5fa5", {
        "mesolve", "mcsolve", "sesolve", "stochastic", "bloch_redfield",
        "nonmarkov", "floquet", "essolve", "correlation", "steadystate",
        "rhs_generate", "propagator", "eseries", "hsolve", "rcsolve",
        "scattering", "piqs", "pdpsolve",
    }),
    # Options and settings
    ("#043c6b", {"settings", "configrc", "solver"}),
    # Visualisation
    ("#3f8fd2", {
        "bloch", "bloch3d", "sphereplot", "orbital", "visualization", "wigner",
        "distributions", "tomography", "topology",
    }),
    # Operators
    ("#00ae68", {
        "operators", "superoperator", "superop_reps", "subsystem_apply",
    }),
    # States
    ("#007143", {
        "states", "continuous_variables", "qstate", "random_objects",
        "three_level_atom",
    }),
    # QIP
    ("#36d695", {"measurement"}),
    # Metrics and distance measures
    ("#ff4500", {"entropy", "metrics", "countstat", "semidefinite"}),
    # Core
    ("#692102", {
        "qobj", "qobjevo", "expect", "tensor", "partial_transpose", "ptrace",
        "cy", "fastsparse", "interpolate",
    }),
    # Utilities
    ("#bf5730", {
        "fileio", "utilities", "ipynbtools", "sparse", "graph", "simdiag",
        "permute", "demos", "about", "parallel", "version", "testing",
        "hardware_info", "ui", "cite",
    }),
]

# Set of modules that we don't want to include in the output.  Any modules that
# are detected inside `qutip` but are not either in this set or the
# `module_groups` list will generate a warning when the script is run.
modules_ignored = {
    "dimensions",
    "logging_utils",
    "matplotlib_utilities",
    "legacy",
    "qobjevo_codegen",
    "_mkl",
    "cy.pyxbuilder",
    "cy.openmp",
    "cy.graph_utils",
    "cy.inter",
    "cy.cqobjevo",
    "cy.cqobjevo_factor",
    "cy.codegen",
    "cy.br_codegen",
    "cy.ptrace",
}


def _our_tree(module, tree):
    """
    Find the subtree corresponding to this module, creating any necessary
    subtrees along the way.
    """
    our_tree = tree
    cur_name = ""
    for part in module.__name__.split(".")[1:]:
        cur_name = (cur_name + "." + part) if cur_name else part
        if cur_name in modules_ignored:
            return tree
        try:
            our_tree = our_tree[part]
        except KeyError:
            our_tree[part] = {}
            our_tree = our_tree[part]
    return our_tree


def _ignore(module, root):
    if not module.__name__.startswith(root):
        return True
    name = module.__name__[len(root):]
    if name in modules_ignored:
        return True
    while (idx := name.rfind(".")) > 0:
        name = name[:idx]
        if name in modules_ignored:
            return True
    return False


def python_object_tree(module, tree=None, seen=None, root=None, nobjects=0):
    """
    Recursively access every accessible element of the given module, building
    up a complete tree structure where the keys are the parts of the module
    name, and the eventual leaves are public functions and classes defined in
    that particular module (so ignoring any names that leak in from other
    imports).  For example,
        >>> import qutip
        >>> python_object_tree(qutip)
        {
            "mesolve" : {
                "mesolve": <function qutip.mesolve.mesolve(...)>,
            },
            ...
        }
    """
    tree = tree if tree is not None else {}
    seen = seen if seen is not None else set()
    root = root if root is not None else (module.__name__ + ".")
    if module in seen:
        return tree, nobjects
    seen.add(module)
    our_tree = _our_tree(module, tree)
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isroutine(obj):
            object_module = inspect.getmodule(obj)
            if object_module is module:
                if not obj.__name__.startswith("_"):
                    our_tree[obj.__name__] = obj
                    nobjects += 1
                continue
            # Fall through, so we recursively comb through modules.
            obj = object_module
        if inspect.ismodule(obj) and not _ignore(obj, root):
            if obj.__name__.startswith(root):
                _, nobjects =\
                    python_object_tree(obj, tree, seen, root, nobjects)
    # Also do our parent package, if we have one.  In theory it's possible to
    # get into a situation with packages and overzealous use of "del" in init
    # scripts where a submodule may be accessible but its parent isn't.
    parent = ".".join(module.__name__.split(".")[:-1])
    if parent.startswith(root):
        _, nobjects =\
            python_object_tree(sys.modules[parent], tree, seen, root, nobjects)
    return tree, nobjects


def _lookup_color(basename, index, color):
    for i, (color_, modules) in enumerate(module_groups):
        if basename in modules:
            return i, color_
    return index, color


def convert_to_d3_struct(in_tree, name, index=-1, color=None, basename=None):
    out_struct = {}
    children = []
    color_default = "black"
    index, color = _lookup_color(basename, index, color)
    for key, value in in_tree.items():
        nextname = (basename + "." + key) if basename else key
        if isinstance(value, dict):
            out = convert_to_d3_struct(value, key, index, color, nextname)
        else:
            out = {
                "name": key,
                "color": color or color_default,
                "index": index,
            }
        children.append(out)
    if name == "QuTiP" and basename is None:
        # Don't warn for the base case.
        color = color_default
    if color is None:
        modname = "qutip" + (("." + basename) if basename else "")
        warnings.warn("handling unspecified module: " + modname)
    out_struct["name"] = name
    out_struct["color"] = color or color_default
    out_struct["index"] = index
    if children:
        out_struct["children"] = sorted(children, key=lambda x: x["index"])
    return out_struct


if __name__ == "__main__":
    import json

    tree, count = python_object_tree(qutip)
    struct = convert_to_d3_struct(tree, "QuTiP")
    with open("d3_data/qutip.json", "w") as f:
        json.dump(struct, f)
    print(count)
