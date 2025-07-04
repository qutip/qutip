.. _developers:

************
Developers
************

.. plot::
    :context: close-figs
    :include-source: False

    import json
    import urllib.request

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    import PIL


    LINK_CONTRIBUTORS = "https://api.github.com/repos/qutip/qutip/contributors"
    LINK_LOGO = "https://qutip.org/images/logo.png"

    # font properties
    FONT_SIZE = 6
    FONT_FAMILY = "DejaVu Sans"

    # figures properties
    FIGURE_SIZE = 8
    AXIS_SIZE = 50
    FONT_COLOR = "black"
    LOGO_SIZE = 40
    LOGO_TRANSPARENCY = 0.5

    # load the list of contributors from qutip/qutip repo
    url_object = urllib.request.urlopen(LINK_CONTRIBUTORS)
    list_contributors = json.loads(url_object.read())
    qutip_contributors = [element["login"] for element in list_contributors]
    qutip_contributors = [s.lower() for s in qutip_contributors]
    text = " ".join(qutip_contributors)

    # load the QuTiP logo
    img = PIL.Image.open(urllib.request.urlopen(LINK_LOGO))

    # code below was inspired in the following link:
    # https://github.com/dynamicwebpaige/nanowrimo-2021/blob/main/15_VS_Code_contributors.ipynb

    n = 100
    A = np.linspace(np.pi, n * 2 * np.pi, 10_000)
    R = 5 + np.linspace(np.pi, n * 2 * np.pi, 10_000)
    T = np.stack([R * np.cos(A), R * np.sin(A)], axis=1)
    dx = np.cos(A) - R * np.sin(A)
    dy = np.sin(A) + R * np.cos(A)
    O = np.stack([-dy, dx], axis=1)
    O = O / (np.linalg.norm(O, axis=1)).reshape(len(O), 1)

    L = np.zeros(len(T))
    np.cumsum(np.sqrt(((T[1:] - T[:-1]) ** 2).sum(axis=1)), out=L[1:])

    path = TextPath(
        (0, 0), text,
        size=FONT_SIZE,
        prop=FontProperties(family=FONT_FAMILY),
    )

    vertices = path.vertices
    codes = path.codes

    Vx, Vy = vertices[:, 0], vertices[:, 1]
    X = np.interp(Vx, L, T[:, 0]) + Vy * np.interp(Vx, L, O[:, 0])
    Y = np.interp(Vx, L, T[:, 1]) + Vy * np.interp(Vx, L, O[:, 1])
    vertices = np.stack([X, Y], axis=-1)

    path = Path(vertices, codes, closed=False)

    # creating figure
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    patch = PathPatch(path, facecolor=FONT_COLOR, linewidth=0)
    ax.add_artist(patch)
    ax.set_xlim(-AXIS_SIZE, AXIS_SIZE), ax.set_xticks([])
    ax.set_ylim(-AXIS_SIZE, AXIS_SIZE), ax.set_yticks([])

    # add qutip logo
    ax.imshow(img, alpha=LOGO_TRANSPARENCY,
              extent=[-LOGO_SIZE,LOGO_SIZE, -LOGO_SIZE, LOGO_SIZE])

.. _developers-lead:

Lead Developers
===============

- `Alex Pitchford <https://github.com/ajgpitch>`_
- `Nathan Shammah <https://nathanshammah.com/>`_
- `Shahnawaz Ahmed <http://sahmed.in/>`_
- `Neill Lambert <https://github.com/nwlambert>`_
- `Eric Giguère <https://github.com/Ericgig>`_
- `Boxi Li <https://github.com/BoxiLi>`_
- `Simon Cross <http://hodgestar.za.net/>`_
- `Asier Galicia <https://github.com/AGaliciaMartinez>`_
- `Paul Menczel <www.menczel.net>`_

Past Lead Developers
====================

- `Robert Johansson <https://jrjohansson.github.io/research.html>`_ (RIKEN)
- `Paul Nation <https://www.korea.ac.kr/>`_ (Korea University)
- `Chris Granade <https://www.cgranade.com>`_
- `Arne Grimsmo <https://www.sydney.edu.au/science/about/our-people/academic-staff/arne-grimsmo.html>`_
- `Jake Lishman <https://binhbar.com>`_


.. _developers-contributors:

Contributors
============

.. note::

  Anyone is welcome to contribute to QuTiP.
  If you are interested in helping, please let us know!


- Abhisek Upadhyaya
- adria.labay
- Adriaan
- AGaliciaMartinez
- alan-nala
- Alberto Mercurio
- alex
- Alexander Pitchford
- Alexios-xi
- Amit
- Andrey Nikitin
- Andrey Rakhubovsky
- Anna Naden
- anonymousdouble
- Anto Luketina
- Antonio Andrea Gentile
- Anubhav Vardhan
- Anush Venkatakrishnan
- Arie van Deursen
- Arne Grimsmo
- Arne Hamann
- Aryaman Kolhe
- Ashish Panigrahi
- Asier Galicia Martinez
- awkwardPotato812
- Ben Bartlett
- Ben Criger
- Ben Jones
- Bo Yang
- Bogdan Reznychenko
- Boxi Li
- CamilleLCal
- Canoming
- christian512
- christian512
- Christoph Gohlke
- Christopher Granade
- Craig Gidney
- Daniel Weiss
- Danny
- davidschlegel
- Denis Vasilyev
- dependabot[bot]
- dev-aditya
- DnMGalan
- Dominic Meiser
- Drew Parsons
- drodper
- dweigand
- Edward Thomas
- Élie Gouzien
- eliegenois
- Emi
- EmilianoG-byte
- Eric Giguère
- Eric Hontz
- essence-of-waqf
- Felipe Bivort Haiek
- fhenneke
- fhopfmueller
- Florestan Ziem
- Florian Hopfmueller
- gadhvirushiraj
- Gaurav Saxena
- gecrooks
- Gerardo Jose Suarez
- Gilbert Shih
- Harry Adams
- Harsh Khilawala
- HGSilveri
- Hristo Georgiev
- Ivan Carvalho
- Jake Lishman
- jakobjakobson13
- Javad Noorbakhsh
- Jevon Longdell
- Johannes Feist
- Jon Crall
- Jonas Hoersch
- Jonas Neergaard-Nielsen
- Jonathan A. Gross
- Joseph Fox-Rabinovitz
- Julian Iacoponi
- Kevin Fischer
- Kosuke Mizuno
- kwyip
- L K Livingstone
- Lajos Palanki
- Laurence Stant
- Laurent AJDNIK
- Leo_am
- Leonardo Assis
- Louis Tessler
- Lucas Verney
- Maggie
- Mahdi Aslani
- maij
- Marco David
- Marek
- marekyggdrasil
- Mark Johnson
- Markus Baden
- Martín Sande
- Mateo Laguna
- Matt
- Matthew O'Brien
- Matthew Treinish
- mcditooss
- Mehdi Aslani
- Michael Goerz
- Michael V. DePalatis
- Moritz Oberhauser
- MrRobot2211
- Nathan Shammah
- Neill Lambert
- Nicolas Quesada
- Nikhil Harle
- Nikolas Tezak
- Nithin Ramu
- obliviateandsurrender
- owenagnel
- Paul
- Paul Menczel
- Paul Nation
- Peter Kirton
- Philipp Schindler
- PierreGuilmin
- Pieter Eendebak
- Piotr Migdal
- PositroniumJS
- Purva Thakre
- quantshah
- Rajath Shetty
- Rajiv-B
- Ray Ganardi
- Reinier Heeres
- Richard Brierley
- Rita Abani
- Robert Johansson
- Rochisha Agarwal
- rochisha0
- ruffa
- Rushiraj Gadhvi
- Sam Griffiths
- Sam Wolski
- Samesh Lakhotia
- Sampreet Kalita
- sbisw002
- Sebastian Krämer
- Shahnawaz Ahmed
- Sidhant Saraogi
- Simon Cross
- Simon Humpohl
- Simon Whalen
- SJUW
- Srinidhi P V
- Stefan Krastanov
- tamakoshi
- tamakoshi2001
- Tanya Garg
- Tarun Raheja
- Thomas Walker
- Tobias Schmale
- trentfridey
- valanm22
- Viacheslav Ostroukh
- Victory Omole
- vikas-chaudhary-2802
- Vlad Negnevitsky
- Vladimir Vargas-Calderón
- Wikstahl
- WingCode
- Wojciech Rzadkowski
- Xavier Spronken
- Xiaodong Qi
- Xiaoliang Wu
- xspronken
- Yariv Yanay
- Yash-10
- YouWei Zhao
- Yuji TAMAKOSHI
- yulanl22
- yuri@FreeBSD
