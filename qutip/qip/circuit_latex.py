# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without 
#    modification, are permitted provided that the following conditions are 
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
import os
import numpy as np
import subprocess as sub
from qutip.qip.qcircuit_latex import _qcircuit_latex_min

_latex_template = r"""
\documentclass{standalone}
%s
\begin{document}
\Qcircuit @C=1cm @R=1cm {
%s}
\end{document}
"""

def _latex_compile(code, filename="qcirc", format="png"):
    """
    Requires: pdflatex, pdfcrop, pdf2svg, imagemagick (convert)
    """
    os.system("rm -f %s.tex %s.pdf %s.png" % (filename, filename, filename))        

    with open(filename + ".tex", "w") as file:
        file.write(_latex_template % (_qcircuit_latex_min, code))

    os.system("pdflatex -interaction batchmode %s.tex" % filename)
    os.system("rm -f %s.aux %s.log" % (filename, filename))        
    os.system("pdfcrop %s.pdf %s-tmp.pdf" % (filename, filename))
    os.system("mv %s-tmp.pdf %s.pdf" % (filename, filename))        

    if format == 'png':
        os.system("convert -density %s %s.pdf %s.png" % (100, filename, 
                                                         filename))
        with open("%s.png" % filename, "rb") as f:
            result = f.read()
    else:
        os.system("pdf2svg %s.pdf %s.svg" % (filename, filename))
        with open("%s.svg" % filename) as f:
            result = f.read()

    return result


