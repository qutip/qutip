"""
Citation generator for QuTiP
"""
import os

__all__ = ['cite']


def cite(save=False, path=None):
    """
    Citation information and bibtex generator for QuTiP

    Parameters
    ----------
    save: bool
        The flag specifying whether to save the .bib file.

    path: str
        The complete directory path to generate the bibtex file.
        If not specified then the citation will be generated in cwd
    """
    citation = """\
@misc{qutip5,
  title = {{QuTiP} 5: The Quantum Toolbox in {Python}},
  author = {Lambert, Neill and Giguère, Eric and Menczel, Paul and Li, Boxi
    and Hopf, Patrick and Suárez, Gerardo and Gali, Marc and Lishman, Jake
    and Gadhvi, Rushiraj and Agarwal, Rochisha and Galicia, Asier
    and Shammah, Nathan and Nation, Paul D. and Johansson, J. R.
    and Ahmed, Shahnawaz and Cross, Simon and Pitchford, Alexander
    and Nori, Franco},
  year={2024},
  eprint={2412.04705},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2412.04705},
  doi={10.48550/arXiv.2412.04705},
}"""
    print(citation)

    if not path:
        path = os.getcwd()

    if save:
        filename = "qutip.bib"
        with open(os.path.join(path, filename), 'w') as f:
            f.write("\n".join(citation))


if __name__ == "__main__":
    cite()
