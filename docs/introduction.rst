Introduction
============
lmfit-additional-models: An XPS Fitting Extension for lmfit
___________________________________________________________

Welcome to lmfit-additional-models, a small Python package designed as an extension for the popular `lmfit package <https://lmfit.github.io/lmfit-py/intro.html#>`_
, specifically tailored for X-ray Photoelectron Spectroscopy (XPS) data analysis.

While lmfit provides simple tools to build complex fitting models for non-linear least-squares problems and applies these models to real data, as well as introduces several built-in models, lmfit-additional-models acts as an extension to lmfit designed for XPS data analysis.
lmfit-additional-models provides a comprehensive set of functions and models that facilitate the fitting of XPS spectra.
In particular, lmfit-additional-models provides several models, which use the convolution of a gaussian with model functions of the lmfit-package.

In XPS experiments, the measured data is always a convolution of the physics in the sample and a gaussian due to experimental broadening.
The Gaussian distribution thereby serves as an approximation for the convolution of three distinct Gaussian broadening functions that play a role in the photoemission process:

#. Broadening caused by the excitation source.
#. Broadening resulting from thermal broadening and vibration modes (phonon broadening, depending on the material).
#. Broadening introduced by the analyzer/spectrometer.

In addition to models for fittig signals in XPS data, lmfit-additional-models introduces several background models which can be included in the fit model for fitting the data rather then substracting a precalculated background.
This is the so-called active approach as suggested by `A. Herrera-Gomez <https://doi.org/10.1002/sia.5453>`_ and generally leads to better fit results.

Besides the mentioned models, lmfit-additional-models provides all functions on which the models are based and functions to substract the Tougaard and Shirley background before fitting the data.
In addition, convolution functions based on numpy are provided, allowing users to generate `lmfit CompositeModels <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_ using self-choosen model functions.

Installation
------------

To install lmfit-additional-models, simply use pip::

    $ pip install lmfit-additional-models

Getting Started
---------------

For a quick start guide and examples, please refer to the `Getting Started`_-section as well as the `lmfit`_ -documentation.

.. _lmfit: https://lmfit.github.io/lmfit-py/intro.html
.. _Getting Started: https://lmfit-additional-models.readthedocs.io/en/gh-pages/usage.html


.. note::
    lmfit-additional-models is still under active development, and we welcome your feedback and contributions on the `GitHub repository`_. In addition, we are currently working on the documentation. If you have any questions, please open an issue or discussion on our `GitHub repository`_ .

.. _GitHub repository: https://github.com/Julian-Hochhaus/lmfit-additional-models


