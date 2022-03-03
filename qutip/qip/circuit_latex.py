import collections
import functools
import os
import sys
import shutil
import subprocess
import tempfile
import warnings

_latex_template = r"""
\documentclass{standalone}
\usepackage[braket]{qcircuit}
\renewcommand{\qswap}{*=<0em>{\times}}
\begin{document}
\Qcircuit @C=1cm @R=1cm {
%s}
\end{document}
"""


def _run_command(command, *args, **kwargs):
    """
    Run a command with stdout explicitly thrown away, raising
    `RuntimeError` with the system error message
    if the command returned a non-zero exit code.
    """
    try:
        return subprocess.run(
            command, *args,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            **kwargs,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode(sys.stderr.encoding)) from None


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
                    file.write(_latex_template % code)
                try:
                    _run_command((_pdflatex, '-interaction', 'batchmode',
                                  filename))
                except RuntimeError as e:
                    message = (
                        "pdflatex failed."
                        " Perhaps you do not have it installed, or you are"
                        " missing the LaTeX package 'qcircuit'."
                    )
                    raise RuntimeError(message) from e
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
