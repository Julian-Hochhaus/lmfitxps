<p align="center">
  <img src="src/logos/logo_large.png" alt="lmfitxps">
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.8181379"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8181379.svg" alt="DOI"></a>
  <a href="https://lmfitxps.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/lmfitxps/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://pypi.org/project/lmfitxps/"><img src="https://img.shields.io/pypi/v/PACKAGE?label=pypi%20lmfitxps" alt="PyPI"></a>
  <a href="https://static.pepy.tech/badge/lmfitxps"><img src="https://static.pepy.tech/badge/lmfitxps" alt="Downloads"></a>
  <a href="https://opensource.org/licenses/"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <a href="https://en.wikipedia.org/wiki/Free_and_open-source_software"><img src="https://img.shields.io/badge/FOSS-100%25-green.svg?style=flat" alt="FOSS: 100%"></a>
</p>



## Introduction
Welcome to lmfitxps, a small Python package designed as an extension for the popular lmfit package , specifically tailored for X-ray Photoelectron Spectroscopy (XPS) data analysis.

While lmfit provides simple tools to build complex fitting models for non-linear least-squares problems and applies these models to real data, as well as introduces several built-in models, lmfitxps acts as an extension to lmfit designed for XPS data analysis. lmfitxps provides a comprehensive set of functions and models that facilitate the fitting of XPS spectra. In particular, lmfitxps provides several models, which use the convolution of a gaussian with model functions of the lmfit-package.

In addition to models for fittig signals in XPS data, lmfitxps introduces several background models which can be included in the fit model for fitting the data rather then substracting a precalculated background. This is the so-called active approach as suggested by A. Herrera-Gomez and generally leads to better fit results.
For further details, please refer to the documentation of [lmfitxps](https://lmfitxps.readthedocs.io/en/latest/index.html) and [lmfit](https://lmfit.github.io/lmfit-py/index.html)! 

## Installation
To install lmfitxps, simply use pip:

 `pip install lmfitxps`

 ## How to cite
 [![DOI](https://zenodo.org/badge/642726930.svg)](https://zenodo.org/badge/latestdoi/642726930)


