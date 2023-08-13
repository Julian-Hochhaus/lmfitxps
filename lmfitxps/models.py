import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from .lineshapes import singlett, dublett, fermi_edge, convolve, fft_convolve
from .backgrounds import tougaard, slope, shirley
from lmfit import Model
import lmfit
from lmfit.models import guess_from_peak
from scipy.signal import convolve as sc_convolve

__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "1.3.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"


class ConvGaussianDoniachSinglett(lmfit.model.Model):
    """
    Model of a Doniach dublett profile convoluted with a gaussian.
    See also lmfit->lineshape.gaussian and lmfit->lineshape.doniach.

    """ + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        super().__init__(singlett, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amplitude', value=100, min=0)
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('gamma', value=0.02)
        self.set_param_hint('gaussian_sigma', value=0.2, min=0)
        self.set_param_hint('center', value=100, min=0)
        g_fwhm_expr = '2*{pre:s}gaussian_sigma*1.1774'
        self.set_param_hint('gaussian_fwhm', expr=g_fwhm_expr.format(pre=self.prefix))
        l_fwhm_expr = '{pre:s}sigma*(2+{pre:s}gamma*2.5135+({pre:s}gamma*3.6398)**4)'
        self.set_param_hint('lorentzian_fwhm', expr=l_fwhm_expr.format(pre=self.prefix))

    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        doniach_pars = guess_from_peak(Model(doniach), data, x, negative=False)
        gaussian_sigma = (doniach_pars["sigma"].value)
        params = self.make_params(amplitude=doniach_pars["amplitude"].value, sigma=doniach_pars["sigma"].value,
                                  gamma=doniach_pars["gamma"].value, gaussian_sigma=gaussian_sigma,
                                  center=doniach_pars["center"].value)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ConvGaussianDoniachDublett(lmfit.model.Model):
    """
    Model of a Doniach profile convoluted with a gaussian.
    See also lmfit->lineshape.gaussian and lmfit->lineshape.doniach.
    """ + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        super().__init__(dublett, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amplitude', value=100, min=0)
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('gamma', value=0.02)
        self.set_param_hint('gaussian_sigma', value=0.2, min=0)
        self.set_param_hint('center', value=285)
        self.set_param_hint('soc', value=2.0)
        self.set_param_hint('height_ratio', value=0.75, min=0)
        self.set_param_hint('fct_coster_kronig', value=1, min=0)
        g_fwhm_expr = '2*{pre:s}gaussian_sigma*1.1774'
        self.set_param_hint('gaussian_fwhm', expr=g_fwhm_expr.format(pre=self.prefix))
        l_p1_fwhm_expr = '{pre:s}sigma*(2+{pre:s}gamma*2.5135+({pre:s}gamma*3.6398)**4)'
        l_p2_fwhm_expr = '{pre:s}sigma*(2+{pre:s}gamma*2.5135+({pre:s}gamma*3.6398)**4)*{pre:s}fct_coster_kronig'
        self.set_param_hint('lorentzian_fwhm_p1', expr=l_p1_fwhm_expr.format(pre=self.prefix))
        self.set_param_hint('lorentzian_fwhm_p2', expr=l_p2_fwhm_expr.format(pre=self.prefix))

    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        doniach_pars = guess_from_peak(Model(doniach), data, x, negative=False)
        gaussian_sigma = (doniach_pars["sigma"].value)
        soc_guess = 1
        params = self.make_params(amplitude=doniach_pars['amplitude'].value, sigma=doniach_pars["sigma"].value,
                                  gamma=doniach_pars["gamma"].value, gaussian_sigma=gaussian_sigma,
                                  center=doniach_pars["center"].value, soc=soc_guess, height_ratio=0.75)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class FermiEdgeModel(lmfit.model.Model):
    """
    Model of a ThermalDistribution convoluted with a gaussian.
    See also lmfit->lineshape.gaussian and lmfit->lineshape.thermal_distribution.
    """ + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        super().__init__(fermi_edge, *args, **kwargs)
        # limit several input parameters to positive values
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('kt', value=0.02585, min=0)  # initial value is room temperature
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('center', value=100, min=0)
        self.set_param_hint('amplitude', value=100, min=0)

    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        self.set_param_hint('center', value=np.mean(x), min=min(x), max=max(x))
        self.set_param_hint('kt', value=kb * 300, min=0, max=kb * 1500)
        self.set_param_hint('amplitude', value=(max(data) - min(data)) / 10, min=0, max=(max(data) - min(data)))
        self.set_param_hint('sigma', value=(max(x) - min(x)) / len(x), min=0, max=2)
        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class TougaardBG(lmfit.model.Model):
    __doc__ = """Model of the Tougaard Background
================================

The `TougaardBG` model is based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [1].

In addition to R.Hesse's approach, this model introduces the `extend` parameter, which enhances the agreement between the data and the Tougaard background by extending the data on the high-kinetic energy side (low binding energy side) using the mean value of the rightmost ten intensity values (with regard to kinetic energy scale, binding energy scale vice versa).

The `extend` parameter represents the length of the data extension on the high-kinetic-energy side in eV and defaults to 0.

This approach was found to lead to great convergence empirically with the extend value being in the range of 25eV to 75eV; however, the length of the data extension remains arbitrary and depends on the dataset.

For further details, please refer to the documentation.

The Tougaard background is calculated using:

.. math::

    B(E) = \\int_{E}^{\\infty} \\frac{B \\cdot T}{{(C + C_d^2)^2} + D \\cdot T^2} \\cdot y(E') \\, dE'

where:

    - :math:`B(E)` represents the Tougaard background at energy :math:`E`,
    - :math:`y` is the measured intensity
    - :math:`T` is :math:`E' - E`.
    - :math:`B` parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. Acts as the scaling factor of the Tougaard background model.
    - :math:`C` , :math:`C_d` and :math:`D` are parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_.

To generate the 2-PIESCS loss function, set :math:`C_d` to 1 and :math:`D` to 0. 
Set :math:`C_d=1` and :math:`D` :math:`\!=0` to get the 3-PIESCS loss function.

For further details on the 2-PIESCS loss function, please refer to S.Tougaard [2]_, and for the
3-PIESCS loss function, see S. Tougaard [3]_.

Notes
-----

The `TougaardBG` class inherits from `lmfit.model.Model` and extends it with specific behavior and functionality related to the Tougaard 4-parameter loss function.
    
Args
----
    `independent_vars` : :obj:`list` of :obj:`str`, optional
        Arguments to the model function that are independent variables
        default is ['x']).
    `prefix` : str, optional
        String to prepend to parameter names, needed to add two Models
        that have parameter names in common.
    `nan_policy` : {'raise', 'propagate', 'omit'}, optional
        How to handle NaN and missing values in data. See Notes below.
    `**kwargs` : optional
        Keyword arguments to pass to :class:`Model`.
    
Notes
-----
    1. `nan_policy` sets what to do when a NaN or missing value is seen in
    the data. Should be one of:

        - `'raise'` : raise a `ValueError` (default)
        - `'propagate'` : do nothing
        - `'omit'` : drop missing data

References
----------
.. [1] R. Hesse; R. Denecke (2011). `Improved Tougaard background calculation by introduction of fittable parameters for the inelastic electron scattering cross-section in the peak fit of photoelectron spectra with UNIFIT 2011.`,` 43(12), 1514–1526. doi:10.1002/sia.3746 `
.. [2] Tougaard, S. (1987). "Low energy inelastic electron scattering properties of noble and transition metals" Solid State Communications, 61(9), 547–549. https://doi.org/10.1016/0038-1098(87)90166-9
.. [3] Tougaard, S. (1997), Universality Classes of Inelastic Electron Scattering Cross-sections. Surf. Interface Anal., 25: 137-154. https://doi.org/10.1002/(SICI)1096-9918(199703)25:3<137::AID-SIA230>3.0.CO;2-L
"""

    def __init__(self, *args, **kwargs):
        """
        Initializes the TougaardBG model instance.

        """
        super().__init__(tougaard, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """
        Sets parameter hints for the model.

        The method sets initial values and constraints for the parameters 'B', 'C', 'C_d', 'D', and 'extend'.

        """
        self.set_param_hint('B', value=2886, min=0)
        self.set_param_hint('C', value=1643, min=0)
        self.set_param_hint('C_d', value=1, min=0)
        self.set_param_hint('D', value=1, min=0)
        self.set_param_hint('extend', value=0, vary=False)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        Args
        ----
            `data (array-like)`: Array containing the data (=intensities) to fit.
            `x` (array-like): Array containing the independent variable values.
            `**kwargs`: Arbitrary keyword arguments.

        Returns:
            Args: Initial parameter values for the model.

        Note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(B=2886, C=1643, C_d=1, D=1)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ShirleyBG(lmfit.model.Model):
    __doc__ = """
    Model of the Shirley background for X-ray photoelectron spectroscopy (XPS) spectra. 

    Attributes:
        All attributes are inherited from the lmfit.model.Model class.

    Methods:
        `__init__(*args, **kwargs)`:
            Initializes the ShirleyBG model instance. Calls the `super().__init__()` method of the parent class
            (lmfit.model.Model) and sets parameter hints using _set_paramhints_prefix() method.

        `_set_paramhints_prefix()`:
            Sets parameter hints for the model. Sets initial values and constraints for the parameters 'k' and 'const'.

        `guess(data, x=None, **kwargs)`:
            Generates initial parameter values for the model based on the provided data and optional arguments.

    Note:
        The ShirleyBG class inherits from lmfit.model.Model and extends it with specific behavior and functionality
        related to the Shirley background for XPS spectra.
    """ + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        """
        Initializes the ShirleyBG model instance.

        """
        super().__init__(shirley, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """
        Sets parameter hints for the model.

        The method sets initial values and constraints for the parameters :math:`k` and :math: `const`.

        """
        self.set_param_hint('k', value=0.03, min=0)
        self.set_param_hint('const', value=1000, min=0)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        Args:
            data (array-like): Array containing the data to fit.
            x (array-like): Array containing the independent variable values.
            `**kwargs`: Arbitrary keyword arguments.

        Returns:
            Initial parameter values for the model.

        Note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(k=0.03, const=1000)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class SlopeBG(lmfit.model.Model):
    __doc__ = """
    Model of the Slope background for X-ray photoelectron spectroscopy (XPS) spectra.
    Slope Background implemented as suggested by A. Herrera-Gomez et al in [DOI: 10.1016/j.elspec.2013.07.006].

    Attributes:
        All attributes are inherited from the lmfit.model.Model class.

    Methods:
        `__init__(*args, **kwargs)`:
            Initializes the SlopeBG model instance. Calls the `super().__init__()` method of the parent class
            (lmfit.model.Model) and sets parameter hints using _set_paramhints_prefix() method.

        _set_paramhints_prefix():
            Sets parameter hints for the model. Sets an initial value for the parameter 'k'.

        `guess(data, x=None, **kwargs)`:
            Generates initial parameter values for the model based on the provided data and optional arguments.

    Note:
        The SlopeBG class inherits from lmfit.model.Model and extends it with specific behavior and functionality
        related to the Slope background for XPS spectra.
    """ + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        """
        Initializes the SlopeBG model instance.
        """
        super().__init__(slope, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """
        Sets parameter hints for the model.

        The method sets an initial value for the parameter 'k'.

        """
        self.set_param_hint('k', value=0.01)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        Args:
            data (array-like): Array containing the data to fit.
            x (array-like): Array containing the independent variable values.
            `**kwargs:` Arbitrary keyword arguments.

        Returns:
            Parameters: Initial parameter values for the model.

        Note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(k=0.01)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
