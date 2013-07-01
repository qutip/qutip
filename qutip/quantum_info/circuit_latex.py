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
import os
import numpy as np
import subprocess as sub

def _latex_write(name):
         header=_latex_preamble()
         ending=_latex_ending()
         texfile = open(name+".tex", "w")
         texfile.writelines(header+ending)
         texfile.close()


def _latex_pdf(name):
    CWD=os.getcwd()
    _latex_write(name)
    try:
        process = sub.Popen(['pdflatex', CWD+'/'+name+'.tex'], stdout=sub.PIPE)
        stdout, stderr = process.communicate()
        success=1
    except:
        print('pdflatex binary not found.')
        success=0
    if success:
        try:
            os.remove(name+".log")
            os.remove(name+".aux")
        except:
            pass



def _latex_preamble():
    string = "\\documentclass[class=minimal,border=0pt]{standalone}\n"
    string += "\\usepackage{tikz}\n"
    string += "\\usetikzlibrary{backgrounds,fit,decorations.pathreplacing}\n"
    string += "\\newcommand{\\ket}[1]{\\ensuremath{\left|#1\\right\\rangle}}\n"
    string += '\\begin{document}\n'
    return string

def _latex_ending():
    return '\end{document}'


