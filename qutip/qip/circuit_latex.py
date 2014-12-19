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


_qcircuit_latex_min = r"""
% Q-circuit version 2
% Copyright (C) 2004 Steve Flammia & Bryan Eastin
% Last modified on: 9/16/2011
% License: http://www.gnu.org/licenses/gpl-2.0.html
% Original file: http://physics.unm.edu/CQuIC/Qcircuit/Qcircuit.tex
% Modified for QuTiP on: 5/22/2014
\usepackage{xy}
\xyoption{matrix}
\xyoption{frame}
\xyoption{arrow}
\xyoption{arc}
\usepackage{ifpdf}
\entrymodifiers={!C\entrybox}
\newcommand{\bra}[1]{{\left\langle{#1}\right\vert}}
\newcommand{\ket}[1]{{\left\vert{#1}\right\rangle}}
\newcommand{\qw}[1][-1]{\ar @{-} [0,#1]}
\newcommand{\qwx}[1][-1]{\ar @{-} [#1,0]}
\newcommand{\cw}[1][-1]{\ar @{=} [0,#1]}
\newcommand{\cwx}[1][-1]{\ar @{=} [#1,0]}
\newcommand{\gate}[1]{*+<.6em>{#1} \POS ="i","i"+UR;"i"+UL **\dir{-};"i"+DL **\dir{-};"i"+DR **\dir{-};"i"+UR **\dir{-},"i" \qw}
\newcommand{\meter}{*=<1.8em,1.4em>{\xy ="j","j"-<.778em,.322em>;{"j"+<.778em,-.322em> \ellipse ur,_{}},"j"-<0em,.4em>;p+<.5em,.9em> **\dir{-},"j"+<2.2em,2.2em>*{},"j"-<2.2em,2.2em>*{} \endxy} \POS ="i","i"+UR;"i"+UL **\dir{-};"i"+DL **\dir{-};"i"+DR **\dir{-};"i"+UR **\dir{-},"i" \qw}
\newcommand{\measure}[1]{*+[F-:<.9em>]{#1} \qw}
\newcommand{\measuretab}[1]{*{\xy*+<.6em>{#1}="e";"e"+UL;"e"+UR **\dir{-};"e"+DR **\dir{-};"e"+DL **\dir{-};"e"+LC-<.5em,0em> **\dir{-};"e"+UL **\dir{-} \endxy} \qw}
\newcommand{\measureD}[1]{*{\xy*+=<0em,.1em>{#1}="e";"e"+UR+<0em,.25em>;"e"+UL+<-.5em,.25em> **\dir{-};"e"+DL+<-.5em,-.25em> **\dir{-};"e"+DR+<0em,-.25em> **\dir{-};{"e"+UR+<0em,.25em>\ellipse^{}};"e"+C:,+(0,1)*{} \endxy} \qw}
\newcommand{\multimeasure}[2]{*+<1em,.9em>{\hphantom{#2}} \qw \POS[0,0].[#1,0];p !C *{#2},p \drop\frm<.9em>{-}}
\newcommand{\multimeasureD}[2]{*+<1em,.9em>{\hphantom{#2}} \POS [0,0]="i",[0,0].[#1,0]="e",!C *{#2},"e"+UR-<.8em,0em>;"e"+UL **\dir{-};"e"+DL **\dir{-};"e"+DR+<-.8em,0em> **\dir{-};{"e"+DR+<0em,.8em>\ellipse^{}};"e"+UR+<0em,-.8em> **\dir{-};{"e"+UR-<.8em,0em>\ellipse^{}},"i" \qw}
\newcommand{\control}{*!<0em,.025em>-=-<.2em>{\bullet}}
\newcommand{\controlo}{*+<.01em>{\xy -<.095em>*\xycircle<.19em>{} \endxy}}
\newcommand{\ctrl}[1]{\control \qwx[#1] \qw}
\newcommand{\ctrlo}[1]{\controlo \qwx[#1] \qw}
\newcommand{\targ}{*+<.02em,.02em>{\xy ="i","i"-<.39em,0em>;"i"+<.39em,0em> **\dir{-}, "i"-<0em,.39em>;"i"+<0em,.39em> **\dir{-},"i"*\xycircle<.4em>{} \endxy} \qw}
\newcommand{\qswap}{*=<0em>{\times} \qw}
\newcommand{\multigate}[2]{*+<1em,.9em>{\hphantom{#2}} \POS [0,0]="i",[0,0].[#1,0]="e",!C *{#2},"e"+UR;"e"+UL **\dir{-};"e"+DL **\dir{-};"e"+DR **\dir{-};"e"+UR **\dir{-},"i" \qw}
\newcommand{\ghost}[1]{*+<1em,.9em>{\hphantom{#1}} \qw}
\newcommand{\push}[1]{*{#1}}
\newcommand{\gategroup}[6]{\POS"#1,#2"."#3,#2"."#1,#4"."#3,#4"!C*+<#5>\frm{#6}}
\newcommand{\rstick}[1]{*!L!<-.5em,0em>=<0em>{#1}}
\newcommand{\lstick}[1]{*!R!<.5em,0em>=<0em>{#1}}
\newcommand{\ustick}[1]{*!D!<0em,-.5em>=<0em>{#1}}
\newcommand{\dstick}[1]{*!U!<0em,.5em>=<0em>{#1}}
\newcommand{\Qcircuit}{\xymatrix @*=<0em>}
\newcommand{\link}[2]{\ar @{-} [#1,#2]}
\newcommand{\pureghost}[1]{*+<1em,.9em>{\hphantom{#1}}}
"""
