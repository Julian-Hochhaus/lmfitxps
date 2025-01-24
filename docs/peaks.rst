.. _PeakModels:

Peak-like/Step-like models
==========================

.. include:: note.rst

The following sections documents the peak-like/step-like models implemented as an extension to the `lmfit build-in models <https://lmfit.github.io/lmfit-py/builtin_models.html>`_ .
The models are thereby mostly based on the `lmfit lineshapes module <https://github.com/lmfit/lmfit-py/blob/master/lmfit/lineshapes.py>`_.



:class:`FermiEdgeModel`
_______________________
.. autoclass:: lmfitxps.models.FermiEdgeModel
    :members:
    :member-order: bysource

.. _ConvGaussianDoniachSinglett:

:class:`ConvGaussianDoniachSinglett`
____________________________________
.. autoclass:: lmfitxps.models.ConvGaussianDoniachSinglett
    :members:
    :member-order: bysource

.. _ConvGaussianDoniachDublett:

:class:`ConvGaussianDoniachDublett`
____________________________________
.. autoclass:: lmfitxps.models.ConvGaussianDoniachDublett
    :members:
    :member-order: bysource

.. _fwhm_doniach:

Approximation to the FWHM of Doniach-Sunjic Line Shape
------------------------------------------------------

The Doniach-Sunjic line shape is commonly used in the analysis of X-ray photoelectron spectroscopy (XPS) data, despite having several limitations. Its popularity in the XPS community stems from its ability to accurately represent asymmetric XPS peaks.

However, there are notable challenges associated with the Doniach-Sunjic line shape:

- The area under this line shape is ill-defined and infinite.
- There is no closed formula available for calculating its full width at half maximum (FWHM).

To approximate the FWHM, the following formula is employed:

.. math::

    \text{FWHM}_{DS} = \gamma \cdot \left(2 + \alpha \cdot a + (\alpha \cdot b)^4\right)

In this formula:

- :math:`\gamma` denotes the broadening parameter of the Doniach-Sunjic line shape.
- :math:`\alpha` represents the asymmetry of the line shape.
- The constants :math:`a = 2.5135` and :math:`b = 3.6398` provide an approximation for the FWHM with an error of less than 2% within a reasonable range of asymmetry, specifically when :math:`\alpha < 0.25`.

When :math:`\alpha = 0`, the Doniach-Sunjic line shape simplifies to a Lorentzian line shape, and thus, this approximation formula corresponds to the FWHM of a Lorentzian as well.