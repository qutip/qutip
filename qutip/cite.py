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
    citation = ["@article{qutip2,",
                "doi = {10.1016/j.cpc.2012.11.019},",
                "url = {https://doi.org/10.1016/j.cpc.2012.11.019},",
                "year  = {2013},",
                "month = {apr},",
                "publisher = {Elsevier {BV}},",
                "volume = {184},",
                "number = {4},",
                "pages = {1234--1240},",
                "author = {J.R. Johansson and P.D. Nation and F. Nori},",
                "title = {{QuTiP} 2: A {P}ython framework for the dynamics of open quantum systems},",
                "journal = {Computer Physics Communications}",
                "}",
                "@article{qutip1,",
                "doi = {10.1016/j.cpc.2012.02.021},",
                "url = {https://doi.org/10.1016/j.cpc.2012.02.021},",
                "year  = {2012},",
                "month = {aug},",
                "publisher = {Elsevier {BV}},",
                "volume = {183},",
                "number = {8},",
                "pages = {1760--1772},",
                "author = {J.R. Johansson and P.D. Nation and F. Nori},",
                "title = {{QuTiP}: An open-source {P}ython framework for the dynamics of open quantum systems},",
                "journal = {Computer Physics Communications}",
                "}"]
    print("\n".join(citation))

    if not path:
        path = os.getcwd()

    if save:
        filename = "qutip.bib"
        with open(os.path.join(path, filename), 'w') as f:
            f.write("\n".join(citation))


if __name__ == "__main__":
    cite()
