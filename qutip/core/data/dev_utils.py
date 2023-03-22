"""
Tools to help with data layer development.
"""

from collections import defaultdict
from qutip.core import data as _data

def all_specialisation():
    """
    Create a dict of all specialisations of the dispatcher sorted in number of
    data layer inputs.
    """
    all_dispatchers = qt.data.to.dispatchers
    dispatchers = defaultdict(list)
    for dispatcher in all_dispatchers:
        dispatchers[len(dispatcher.inputs)].append(dispatcher)
    return dict(sorted(dispatchers.items()))

def _bool2str(entry):
    if isinstance(entry, str):
        return entry
    elif entry:
        return "True"
    else:
        return "False"

def _print_table(lines, title):
    if len(lines) == 1: return
    print()
    print(title)
    for line in lines:
        print(f"|{line[0]:<20}|" + "|".join(
            f"{_bool2str(entry):^10}" for entry in line[1:]) + "|"
        )

def _extract_specialisation_per_type(N):
    dispatchers = all_specialisation()
    all_layers = list(qt.data.to.dtypes)
    lines_output = [
        ("Dispatched function",) + tuple(dtype.__name__ for dtype in all_layers)
    ]
    lines_no_output = [
        ("Dispatched function",) + tuple(dtype.__name__ for dtype in all_layers)
    ]
    for spec in dispatchers[N]:
        if spec.output:
            lines_output += [
                (spec.__name__,)
                + tuple((dtype,)*(N+1) in spec._specialisations for dtype in all_layers)
            ]
        else:
            lines_no_output += [
                (spec.__name__,)
                + tuple((dtype,)*N in spec._specialisations for dtype in all_layers)
            ]
    return lines_output, lines_no_output

def specialisation_table():
    """
    Print tables for each dispatched function showing which data layer have a
    specialisation for it. Only pure specialisations are shown, functions
    taking mulitiple input types are not shown.

    Output is sorted by number of data layer input.
    """
    lines_output, lines_no_output = _extract_specialisation_per_type(0)
    _print_table(lines_output, "Output only specialisation")

    lines_output, lines_no_output = _extract_specialisation_per_type(1)
    _print_table(lines_output, "Unitary specialisation")
    _print_table(lines_no_output, "Matrix -> scalar specialisation")

    lines_output, lines_no_output = _extract_specialisation_per_type(2)
    _print_table(lines_output, "Binary specialisation")
    _print_table(lines_no_output, "Binary measurement specialisation")

    lines_output, lines_no_output = _extract_specialisation_per_type(3)
    _print_table(lines_output, "Ternary specialisation")
    _print_table(lines_no_output, "Ternary measurement specialisation")
