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
.. _lmfit.model.Model: https://lmfit.github.io/lmfit-py/model.html#