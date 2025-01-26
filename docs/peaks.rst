.. _PeakModels:

Peak-like/Step-like models
==========================
.. index:: Peak-Models, Fermi edge, Singlett, Dublett

The following sections documents the peak-like/step-like models implemented as an extension to the `lmfit build-in models <https://lmfit.github.io/lmfit-py/builtin_models.html>`_ .
The models are thereby mostly based on the `lmfit lineshapes module <https://github.com/lmfit/lmfit-py/blob/master/lmfit/lineshapes.py>`_.


.. _FermiEdgeModel:

:py:class:`FermiEdgeModel`
__________________________

.. autoclass:: lmfitxps.models.FermiEdgeModel
    :members:
    :member-order: bysource
    :show-inheritance:
    :exclude-members: guess, __init__, _set_paramhints_prefix

.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.

.. _ConvGaussianDoniachSinglett:

:py:class:`ConvGaussianDoniachSinglett`
_______________________________________

.. autoclass:: lmfitxps.models.ConvGaussianDoniachSinglett
    :member-order: bysource
    :exclude-members: guess, __init__, _set_paramhints_prefix

.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.

.. _ConvGaussianDoniachDublett:

:py:class:`ConvGaussianDoniachDublett`
______________________________________

.. autoclass:: lmfitxps.models.ConvGaussianDoniachDublett
    :member-order: bysource
    :exclude-members: guess, __init__, _set_paramhints_prefix

.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.
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