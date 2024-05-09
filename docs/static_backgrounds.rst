Static backgrounds
==================
.. include:: note.rst


The naming of the following two background functions as "static backgrounds" might lead to some confusion. The functions :ref:`shirley_calculate` and :ref:`tougaard_calculate` are referred to as static simply because they aren't designed to be directly integrated into the fitting model itself.

Because these functions are not part of the fit model, they need to be applied separately to the data, removing the background before applying the Levenberg–Marquardt algorithm through lmfit. Unlike the approach in :ref:`BGModels`, where the background is dynamically adjusted and optimized at each iteration step of model optimization, here the background has to be removed before the fitting process begins.

However, these static backgrounds are implemented iteratively, allowing their scaling parameters to be optimized for the input dataset.

Especially in cases where the fitting model is complex, applying these static background functions to the dataset can aid in approximating suitable starting parameters for the background within the fit model.

.. _shirley_calculate:

:py:func:`shirley_calculate`
____________________________


.. autofunction:: lmfitxps.backgrounds.shirley_calculate

.. _tougaard_calculate:

:py:func:`tougaard_calculate`
_____________________________

.. autofunction:: lmfitxps.backgrounds.tougaard_calculate
.. _lmfit.model.Model: https://lmfit.github.io/lmfit-py/model.html#


References
__________
.. [1] Hesse, R., Denecke, R. (2011). Improved Tougaard background calculation by introduction of fittable parameters for the inelastic electron scattering cross-section in the peak fit of photoelectron spectra with UNIFIT 2011.,43(12), 1514–1526. https://doi.org/10.1002/sia.3746
.. [2] Mudd, J. (2011). Igor procedure for subtracting XPS backgrounds. https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/surface/people/james_mudd/igor/
.. [3] Tougaard, S. (1987). Low energy inelastic electron scattering properties of noble and transition metals. Solid State Communications, 61(9), 547–549. https://doi.org/10.1016/0038-1098(87)90166-9
.. [4] Tougaard, S. (1997). Universality Classes of Inelastic Electron Scattering Cross-sections. Surf. Interface Anal., 25: 137-154. https://doi.org/10.1002/(SICI)1096-9918(199703)25:3<137::AID-SIA230>3.0.CO;2-L
.. [5] O'Donnell, K., (2013) Implementation of the auto-Shirley background. https://github.com/kaneod/physics/blob/master/python/specs.py
.. [6] Tougaard, S. (2021). Practical guide to the use of backgrounds in quantitative XPS. Journal of Vacuum Science & Technology A; 39 (1): 011201. https://doi.org/10.1116/6.0000661