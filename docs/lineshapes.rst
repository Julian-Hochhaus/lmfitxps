Model functions
----------------
.. include:: note.rst

In the following subsections, all used model functions which are used for the lmfit models in :ref:`BGModels` as well as in :ref:`PeakModels` are documented.

Lineshapes
~~~~~~~~~~

.. autofunction:: lmfitxps.lineshapes.singlett
.. autofunction:: lmfitxps.lineshapes.dublett
.. autofunction:: lmfitxps.lineshapes.fermi_edge

Background model functions
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: lmfitxps.backgrounds.shirley
.. autofunction:: lmfitxps.backgrounds.slope
.. autofunction:: lmfitxps.backgrounds.tougaard_closure


References
~~~~~~~~~~
.. [1] Hesse, R., Denecke, R. (2011). Improved Tougaard background calculation by introduction of fittable parameters for the inelastic electron scattering cross-section in the peak fit of photoelectron spectra with UNIFIT 2011.,43(12), 1514–1526. https://doi.org/10.1002/sia.3746
.. [2] Tougaard, S. (1987). Low energy inelastic electron scattering properties of noble and transition metals. Solid State Communications, 61(9), 547–549. https://doi.org/10.1016/0038-1098(87)90166-9
.. [3] Tougaard, S. (1997). Universality Classes of Inelastic Electron Scattering Cross-sections. Surf. Interface Anal., 25: 137-154. https://doi.org/10.1002/(SICI)1096-9918(199703)25:3<137::AID-SIA230>3.0.CO;2-L
.. [4] Herrera-Gomez, A., Bravo-Sanchez,M., Aguirre-Tostado, F.S., Vazquez-Lepe, M.O. (2013) The slope-background for the near-peak regimen of photoemission spectra, Journal of Electron Spectroscopy and Related Phenomena, (189), 76-80. https://doi.org/10.1016/j.elspec.2013.07.006.
.. [5] Shirley, D. A. (1972). High-Resolution X-Ray photoemission spectrum of the valence bands of gold. Physical Review, 5(12), 4709–4714. https://doi.org/10.1103/physrevb.5.4709
.. [6] Jansson, C., Tougaard, S., Beamson, G., Briggs, D., Davies, S.F., Rossi, A., Hauert, R., Hobi, G., Brown, N.M.D., Meenan, B.J., Anderson, C.A., Repoux, M., Malitesta, C. and Sabbatini, L. (1995), Intercomparison of algorithms for background correction in XPS. Surf. Interface Anal., 23: 484-494. https://doi.org/10.1002/sia.740230708
