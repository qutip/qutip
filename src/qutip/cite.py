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
  title = {QuTiP 5: The Quantum Toolbox in {Python}},
  author = {
    Lambert, Neill and Gigu{`e}re, Eric and Menczel, Paul and Li, Boxi and
    Hopf, Patrick and Su{'a}rez, Gerardo and Gali, Marc and Lishman, Jake and
    Gadhvi, Rushiraj and Agarwal, Rochisha and Galicia, Asier and Shammah, Nathan and
    Nation, Paul and Johansson, J. R. and Ahmed, Shahnawaz and Cross, Simon and
    Pitchford, Alexander and Nori, Franco
  },
  journal = {Physics Reports},
  volume = {1153},
  pages = {1-62},
  year = {2026},
  issn = {0370-1573},
  doi = {10.1016/j.physrep.2025.10.001},
  url = {https://www.sciencedirect.com/science/article/pii/S0370157325002704},
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
