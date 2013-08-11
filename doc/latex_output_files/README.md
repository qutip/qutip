Contents
-------------------------------------------
latex_preamble.tex contains the formatting options for 
the QuTiP latex docs file.

Copy the sphinx.sty and sphinxmemoir.cls files into the 
QuTiP docs latex directory before compiling.

Instructions
-------------------------------------------

The call to:

\usepackage{times}
\usepackage[Bjarne]{fncychap}

must be manually removed before compilation due to
an options clash.