Introduction
============
Welcome to ``lmfitxps``, a small Python package designed as an extension for the popular `lmfit package <https://lmfit.github.io/lmfit-py/intro.html#>`_
, specifically tailored for X-ray Photoelectron Spectroscopy (XPS) data analysis.

While ``lmfit`` provides simple tools to build complex fitting models for non-linear least-squares problems and applies these models to real data, as well as introduces several built-in models, ``lmfitxps`` acts as an extension to ``lmfit`` designed for XPS data analysis.
``lmfitxps`` provides a comprehensive set of functions and models that facilitate the fitting of XPS spectra.

Although ``lmfit`` already provides several useful models for fitting XPS data, it often proves insufficient in adequately representing experimental XPS data out of the box. In the context of XPS experiments, the observed data is a convolution of both the sample's underlying physical properties and a Gaussian component arising from experimental broadening.

This Gaussian distribution serves as an effective approximation for the convolution of three distinct Gaussian broadening functions, each of which contributes to the complex interplay inherent in the photoemission process:

#. Broadening caused by the excitation source.
#. Broadening resulting from thermal broadening and vibration modes (phonon broadening, depending on the material).
#. Broadening introduced by the analyzer/spectrometer.

For further details, please refer to, for example, the `Practical guide for curve fitting in x-ray photoelectron spectroscopy`_ by G.H. Major et al.

.. _Practical guide for curve fitting in x-ray photoelectron spectroscopy: https://pubs.aip.org/avs/jva/article/38/6/061203/1023652/Practical-guide-for-curve-fitting-in-x-ray

``lmfitxps`` therefore provides convolution functions based on scipy's and numpy's convolution functions to enable the user to build custom `lmfit CompositeModels <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_ using convolution of models.
In addition, ``lmfitxps`` provides several pre-build models, which use the convolution of a gaussian with model functions of the lmfit-package and provides the user with the following models:

.. table:: Peak models
   :widths: 35 65

   +-------------------------------------------+------------------------------------------------------------+
   | Model                                     | Description                                                |
   +===========================================+============================================================+
   |                                           | Convolution of a gaussian with a doniach lineshape used to |
   |``ConvGaussianDoniachSinglett``            | fit singlett XPS peaks such as *s-orbitals*.               |
   |                                           |                                                            |
   +-------------------------------------------+------------------------------------------------------------+
   |                                           | Convolution of a gaussian with a pair of doniach lineshapes|
   |``ConvGaussianDoniachDublett``             | used to fit dublett XPS peaks such as *p-, d-, f-orbitals*.|
   |                                           |                                                            |
   +-------------------------------------------+------------------------------------------------------------+
   |                                           | Convolution of a gaussian with a Fermi Dirac Step function |
   |``FermiEdgeModel``                         | using the thermal distribution lineshape of lmfit.         |
   |                                           |                                                            |
   +-------------------------------------------+------------------------------------------------------------+





In addition to models for fitting signals in XPS data, ``lmfitxps`` introduces several background models which can be included in the fit model for fitting the data rather then substracting a precalculated background.
This is the so-called active approach as suggested by `A. Herrera-Gomez <https://doi.org/10.1002/sia.5453>`_ and generally leads to better fit results.
The available background models are:

.. table:: Background models
   :widths: 25 75

   +-------------------------------------------+------------------------------------------------------------+
   | Model                                     | Description                                                |
   +===========================================+============================================================+
   |    ``ShirleyBG``                          | The commonly used step-like shirley background.            |
   |                                           |                                                            |
   +-------------------------------------------+------------------------------------------------------------+
   |       ``TougaardBG``                      | The Tougaard background based on the four-parameter loss   |
   |                                           | function (4-PIESCS) as suggested by                        |
   |                                           | `R.Hesse <https://doi.org/10.1002/sia.3746>`_ .            |
   +-------------------------------------------+------------------------------------------------------------+
   |                                           | Calculates a sloping background                            |
   |``SlopeBG``                                |                                                            |
   |                                           |                                                            |
   +-------------------------------------------+------------------------------------------------------------+

.. _R.Hesse: https://doi.org/10.1002/sia.3746


In addition to the models discussed above, ``lmfitxps`` provides all the underlying functions that serve as the basis for these models. Furthermore, the package includes functions for removing the Tougaard and Shirley background components before performing data fitting.

Installation
------------
Stable version
______________

To install the stable version of ``lmfitxps``, simply use pip::

    $ pip install lmfitxps

If the required packages were not automatically installed during pip installation or are not yet present on your system, please install the following requirements:

    lmfit>=1.1.0
    matplotlib>=3.6
    numpy>=1.19
    scipy>=1.6


Development version
___________________

To install the development version or to contribute to ``lmfitxps``, please clone the GitHub repository:

.. code-block:: sh

   $ git clone https://github.com/Julian-Hochhaus/lmfitxps.git


Getting Started
---------------

For a quick start guide and examples, please refer to the `Getting Started`_-section as well as the `lmfit`_ -documentation.

.. _lmfit: https://lmfit.github.io/lmfit-py/intro.html
.. _Getting Started: https://lmfitxps.readthedocs.io/en/gh-pages/usage.html


.. note::
    lmfitxps is still under active development, and we welcome your feedback and contributions on the `GitHub repository`_. In addition, we are currently working on the documentation. If you have any questions, please open an issue or discussion on our `GitHub repository`_ .

.. _GitHub repository: https://github.com/Julian-Hochhaus/lmfitxps


