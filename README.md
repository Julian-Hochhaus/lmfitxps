<p align="center">
  <img src="docs/src/logos/logo_large.png" alt="lmfitxps">
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.8181379"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8181379.svg" alt="DOI"></a>
  <a href="https://lmfitxps.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/lmfitxps/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://badge.fury.io/py/lmfitxps"><img src="https://badge.fury.io/py/lmfitxps.svg" alt="PyPI version" height="18"></a>
  <a href="https://static.pepy.tech/badge/lmfitxps"><img src="https://static.pepy.tech/badge/lmfitxps" alt="Downloads"></a>
  <a href="https://opensource.org/licenses/"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <a href="https://en.wikipedia.org/wiki/Free_and_open-source_software"><img src="https://img.shields.io/badge/FOSS-100%25-green.svg?style=flat" alt="FOSS: 100%"></a>
  <a href="https://app.fossa.com/projects/git%2Bgithub.com%2FJulian-Hochhaus%2Flmfitxps?ref=badge_small" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2FJulian-Hochhaus%2Flmfitxps.svg?type=small"/></a>
</p>



## Introduction
Welcome to lmfitxps, a small Python package designed as an extension for the popular lmfit package, specifically tailored for X-ray Photoelectron Spectroscopy (XPS) data analysis.

While lmfit provides simple tools to build complex fitting models for non-linear least-squares problems and applies these models to real data, as well as introduces several built-in models, `lmfitxps` acts as an extension to lmfit designed for XPS data analysis. `lmfitxps` provides a comprehensive set of functions and models that facilitate the fitting of XPS spectra. In particular, lmfitxps provides several models that use the convolution of a Gaussian with model functions of the limit-package.

In addition to models for fitting signals in XPS data, lmfitxps introduces several background models that can be included in the fit model for fitting the data rather than subtracting a precalculated background. This is the so-called active approach, as suggested by A. Herrera-Gomez, and generally leads to better fit results.

For further details, please refer to the documentation of [lmfitxps](https://lmfitxps.readthedocs.io/en/latest/index.html) and [lmfit](https://lmfit.github.io/lmfit-py/index.html)! 

## Installation
To install `lmfitxps`, simply use pip:

 `pip install lmfitxps`

 ## How to cite
 [![DOI](https://zenodo.org/badge/642726930.svg)](https://zenodo.org/badge/latestdoi/642726930)


## List of Publications using lmfitxps

- A. Kononov, [Fullerene and bismuth clusters on nanostructured oxide films](http://dx.doi.org/10.17877/DE290R-24509), Dissertation (2024)
- K. Teenakul et al. [Treatment of carbon electrodes with Ti3C2Tx MXene coating and thermal method for vanadium redox flow batteries: a comparative study](https://doi.org/10.1039/D4RA01380H) RSC Adv., **14**, 12807-12816 (2024).
- P. Weinert et al. [Structural, chemical, and magnetic investigation of a graphene/cobalt/platinum multilayer system on silicon carbide](http://dx.doi.org/10.1088/1361-6528/ad1d7b) Nanotechnology, **35** 165702 (2024). 
- P. Lamichhane et al. [Investigating the synergy of rapidly synthesized iron oxide predecessor and plasma-gaseous species for dye-removal to reuse water in irrigation](https://doi.org/10.1016/j.chemosphere.2024.143040) Chemosphere, **24** 143040 (2024).
- P. Weinert [Structural, chemical, and magnetic investigation of a graphene/cobalt/platinum multilayer system on silicon carbide : About the formation of magnetic structures in 2D cobalt layers](https://d-nb.info/1328839591) Dissertation (2024).
- P. Schöngrundner [Search for crystalline SiO2 on the wet chemically treated 6H-SiC(0001) surface](https://doi.org/10.34726/HSS.2024.124590) TU Wien, Thesis (2024)  
- J. A. Hochhaus et al. [Structural analysis of Sn on Au(111) at low coverages: Towards the Au2Sn surface alloy with alternating fcc and hcp domains](https://doi.org/10.1038/s41598-025-91733-2) Sci. Rep. **15**, 7953 (2025).
  
Publications that use [LG4X-V2](https://github.com/Julian-Hochhaus/LG4X-V2), a graphical user interface (GUI) for XPS/XAS analysis that heavily utilizes `lmfit` and `lmfitxps`, are also included.

