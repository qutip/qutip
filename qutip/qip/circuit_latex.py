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
import collections
import functools
import os
import shutil
import subprocess
import tempfile
import warnings

_latex_template = r"""
\documentclass{standalone}
%s
\begin{document}
\Qcircuit @C=1cm @R=1cm {
%s}
\end{document}
"""


_run_command = functools.partial(subprocess.run, check=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
_run_command.__doc__ = \
    """
    Run a command with stdout and stderr explicitly thrown away, raising
    `subprocess.CalledProcessError` if the command returned a non-zero exit
    code.
    """


def _force_remove(*filenames):
    """`rm -f`: try to remove a file, ignoring errors if it doesn't exist."""
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


def _test_convert_is_imagemagick():
    """
    Test to see if the `convert` command behaves like we'd expect ImageMagick
    to.  On Windows if ImageMagick is not installed then `convert` may refer to
    a system utility.
    """
    try:
        # Don't use `capture_output` because we're still supporting Python 3.6
        process = subprocess.run(('convert', '-version'),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        return "imagemagick" in process.stdout.decode('utf-8').lower()
    except FileNotFoundError:
        return False


_SPECIAL_CASES = {
    'convert': _test_convert_is_imagemagick,
}


def _find_system_command(names):
    """
    Given a list of possible system commands (as strings), return the first one
    which has a locatable executable form, or `None` if none of them do.  We
    also check some special cases of shadowing (e.g. ImageMagick 6's `convert`
    is also a Windows system utility) to try and catch false-positives.
    """
    for name in names:
        if shutil.which(name) is not None:
            is_valid = _SPECIAL_CASES.get(name, lambda: True)()
            if is_valid:
                return name
    return None


_pdflatex = _find_system_command(['pdflatex'])
_pdfcrop = _find_system_command(['pdfcrop'])


if _pdfcrop is not None:
    def _crop_pdf(filename):
        """Crop the pdf file `filename` in place."""
        temporary = ".tmp." + filename
        _run_command((_pdfcrop, filename, temporary))
        # Windows does not allow renaming to an existing file (but unix does).
        _force_remove(filename)
        os.rename(temporary, filename)
else:
    def _crop_pdf(_):
        # Warn, but do not raise - we can recover from a failed crop.
        warnings.warn("Could not locate system 'pdfcrop':"
                      " image output may have additional margins.")


def _convert_pdf(file_stem):
    """
    'Convert' to pdf: since LaTeX outputs a PDF file, there's nothing to do.
    """
    with open(file_stem + ".pdf", "rb") as file:
        return file.read()


# Record type to hold definitions of possible conversions - this is just for
# reading convenience.
_ConverterConfiguration = collections.namedtuple(
    '_ConverterConfiguration',
    ['file_type', 'dependency', 'executables', 'arguments', 'binary'],
)
CONVERTERS = {"pdf": _convert_pdf}
_MISSING_CONVERTERS = {}
_CONVERTER_CONFIGURATIONS = [
    _ConverterConfiguration('png', 'ImageMagick', ['magick', 'convert'],
                            arguments=('-density', '100'), binary=True),
    _ConverterConfiguration('svg', 'pdf2svg', ['pdf2svg'],
                            arguments=(), binary=False),
]


def _make_converter(configuration):
    """
    Create the actual conversion function of signature
        file_stem: str -> 'T,
    where 'T is data in the format to be converted to.
    """
    which = _find_system_command(configuration.executables)
    if which is None:
        return None
    mode = "rb" if configuration.binary else "r"

    def converter(file_stem):
        """
        Convert a file located in the current directory named `<file_stem>.pdf`
        to an image format with the name `<file_stem>.xxx`, where `xxx` is
        converter-dependent.

        Parameters
        ----------
        file_stem : str
            The basename of the PDF file to be converted.
        """
        in_file = file_stem + ".pdf"
        out_file = file_stem + "." + configuration.file_type
        _run_command((which, *configuration.arguments, in_file, out_file))
        with open(out_file, mode) as file:
            return file.read()
    return converter


for configuration in _CONVERTER_CONFIGURATIONS:
    # Make the converter using a higher-order function, because if we defined a
    # function in the loop, it would be easy to later introduce bugs due to
    # leaky closures over loop variables.
    converter = _make_converter(configuration)
    if converter:
        CONVERTERS[configuration.file_type] = converter
    else:
        _MISSING_CONVERTERS[configuration.file_type] = configuration.dependency


if _pdflatex is not None:
    def image_from_latex(code, file_type="png"):
        """
        Convert the LaTeX `code` into an image format, defined by the
        `file_type`.  Returns a string or bytes object, depending on whether
        the requested type is textual (e.g. svg) or binary (e.g. png).  The
        known file types are in keys in this module's `CONVERTERS` dictionary.

        Parameters
        ----------
        code: str
            LaTeX code representing the circuit to be converted.

        file_type: str ("png")
            The file type that the image should be returned in.


        Returns
        -------
        image: str or bytes
            An encoded version of the image.  Whether the output type is str or
            bytes depends on whether the requested image format is textual or
            binary.
        """
        filename = "qcirc"  # Arbitrary and internal.
        # We do all the image conversion in a temporary directory to prevent
        # leftover files if something goes wrong (or we get a
        # KeyboardInterrupt) during conversion.
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                with open(filename + ".tex", "w") as file:
                    file.write(_latex_template % (_qcircuit_latex_min, code))
                _run_command((_pdflatex, '-interaction', 'batchmode',
                              filename))
                _crop_pdf(filename + ".pdf")
                if file_type in _MISSING_CONVERTERS:
                    dependency = _MISSING_CONVERTERS[file_type]
                    message = "".join([
                        "Could not find system ", dependency, ".",
                        " Image conversion to '", file_type, "'",
                        " is not available."
                    ])
                    raise RuntimeError(message)
                if file_type not in CONVERTERS:
                    raise ValueError("".join(["Unknown output format: '",
                                              file_type, "'."]))
                out = CONVERTERS[file_type](filename)
            finally:
                # Leave the temporary directory before it is removed (necessary
                # on Windows, but it doesn't hurt on POSIX).
                os.chdir(previous_dir)
        return out
else:
    def image_from_latex(*args, **kwargs):
        raise RuntimeError("Could not find system 'pdflatex'.")


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
