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

Approximation to the FWHM of Doniach-Sunjic line shape
------------------------------------------------------
Generally speaking, the Doniach-Sunjic line shape has several limitations with regards to its application in analysing XPS data. However, it is still widely used in the XPS community due to its excellent agreement with asymmetric XPS peaks.
The area of the Doniach-Sunjic line shape is ill-defined and infinite. In addition, no closed formula exists for its FWHM.
To give an approximation to the FWHM, the following formula is used:
    .. math::

    \text{FWHM}_{DS}=\gamma \cdot \left(2+ \alpha \cdot a +(\alpha \cdot b )^4 \right)

Thereby:

    - :math:`\gamma` represents the broadening of the Doniach-Sunjic lineshape,
    - :math:`\alpha` is the asymmetry of the Doniach-Sunjic lineshape,
    - :math:` a=2.5135` and :math:` b=3.6398` were found to approximate the FWHM with an error <2% for reasonable range of the asymmetry being :math:`\alpha<0.25`.

If :math:`\alpha=0`, the Doniach-Sunjic lineshape equals the Lorentzian line shape and the approximation formula equals the FWHM of the Lorentzian as well.