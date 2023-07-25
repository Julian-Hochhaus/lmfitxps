# lmfitxps
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


