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
    __doc__ = """
The TougaardBG model is based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [1]_.

| In addition to R.Hesse's approach, this model introduces the `extend` parameter, which enhances the agreement between the data and the Tougaard background by extending the data on the high-kinetic energy side (low binding energy side) using the mean value of the rightmost ten intensity values (with regard to kinetic energy scale, binding energy scale vice versa).
| The `extend` parameter represents the length of the data extension on the high-kinetic-energy side in eV and defaults to 0. 
| This approach was found to lead to great convergence empirically with the extend value being in the range of 25eV to 75eV; however, the length of the data extension remains arbitrary and depends on the dataset.

For further details, please read more in :ref:`extend_parameter`.

The Tougaard background is calculated using:

.. math::

    B_T(E) = \\int_{E}^{\\infty} \\frac{B \\cdot T}{{(C + C_d \\cdot T^2)^2} + D \\cdot T^2} \\cdot y(E') \\, dE'

where:

    - :math:`B_T(E)` represents the Tougaard background at energy :math:`E`,
    - :math:`y(E')` is the measured intensity at :math:`E'`,
    - :math:`T` is the energy difference :math:`E' - E`.
    - :math:`B` parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. Acts as the scaling factor for the Tougaard background model.
    - :math:`C` , :math:`C_d` and :math:`D` are parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_.

To generate the 2-PIESCS loss function, set :math:`C_d` to 1 and :math:`D` to 0. 
Set :math:`C_d=1` and :math:`D !=`  :math:`0` to get the 3-PIESCS loss function.

For further details on the 2-PIESCS loss function, please refer to S.Tougaard [2]_, and for the
3-PIESCS loss function, see S. Tougaard [3]_.

.. table:: Model-specific available parameters
   :widths: auto

   +-----------+---------------+----------------------------------------------------------------------------------------+
   | Parameters|  Type         | Description                                                                            |
   +===========+===============+========================================================================================+
   | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | B         | :obj:`float`  | B parameter of the 4-PIESCS loss function [1]_.                                        |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | C         | :obj:`float`  | C parameter of the 4-PIESCS loss function [1]_.                                        |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | C_d       | :obj:`float`  | C' parameter of the 4-PIESCS loss function [1]_.                                       |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | D         | :obj:`float`  | D parameter of the 4-PIESCS loss function [1]_.                                        |
   +-----------+---------------+----------------------------------------------------------------------------------------+
   | extend    | :obj:`float`  | Determines, how far the spectrum is extended on the right (in eV).                     |
   +-----------+---------------+----------------------------------------------------------------------------------------+


Hint
----

The `TougaardBG` class inherits from `lmfit.model.Model` and extends it with specific behavior and functionality related to the Tougaard 4-parameter loss function.
    
Args
----
    `independent_vars` : :obj:`list` of :obj:`str`, optional
        Arguments to the model function that are independent variables
        default is ['x']).
    `prefix` : :obj:`str`, optional
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

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable values.
        :type x: array-like
        :param kwargs: Arbitrary keyword arguments.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(B=2886, C=1643, C_d=1, D=1)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ShirleyBG(lmfit.model.Model):
    __doc__ = """
    Model of the Shirley background for X-ray photoelectron spectroscopy (XPS) spectra. 
    This implementation calculates the Shirley background by integrating the step characteristic of the spectrum.
    For further details, please refer to Shirley [6]_ or Jansson et al. [7]_.
    
    The Shirley background is calculated using the following integral:
    
    .. math::
        :label: shirley
        
        B_S(E)=k\\cdot \\int_{E}^{E_{\\text{right}}}\\left[I(E')-I_{\\text{right}}\\right] \\, dE'
        
    The Shirley background is typically calculated iteratively using the following formula:

    .. math::
        :label: shirley2

        B_{S, n}(E) = k_n \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} - B_{S, n-1}(E')] \\, dE'

    The iterative process continues until the difference :math:`B_{S, n}(E) - B_{S, n-1}(E)` becomes smaller than a specified tolerance value :math:`tol`. This approach is implemented in the function referenced as :ref:`shirley_calculate`.

    However, calculating the Shirley background before fitting and calculating it with high precision in each fitting step are not practically meaningful. Instead, the Shirley background is computed according to the equation :math:numref:`shirley` within each iteration of the model optimization. This ensures that the Shirley background is adaptively determined during the fitting process, preserving the iterative concept of its calculation.

    .. table:: Model-specific available parameters
        :widths: auto

        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | Parameters | Type          | Description                                                                                        |
        +============+===============+====================================================================================================+
        | x          | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                                       |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | y          | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                                    |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | k          | :obj:`float`  | Shirley parameter :math:`k`, determines step-height of the Shirley background.                     |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | const      | :obj:`float`  | Constant value added to the step-like Shirley background, often set to :math:`I_{\\text{right}}`.   |
        +------------+---------------+----------------------------------------------------------------------------------------------------+

        
    Hint
    ----
    
    The `ShirleyBG` class inherits from `lmfit.model.Model` and acts as a predefined model for calculating the Shirley background.
        
    Args
    ----
        `independent_vars` : :obj:`list` of :obj:`str`, optional
            Arguments to the model function that are independent variables
            default is ['x']).
        `prefix` : :obj:`str`, optional
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
    
    """

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

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable values.
        :type x: array-like
        :param kwargs: Arbitrary keyword arguments.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(k=0.03, const=1000)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class SlopeBG(lmfit.model.Model):
    __doc__ = """
    Model of the Slope background for X-ray photoelectron spectroscopy (XPS) spectra.
    The Slope Background is implemented as suggested by A. Herrera-Gomez et al in [8]_.
    Hereby, while the Shirley background is designed to account for the difference in background height between the two sides of a peak, the Slope background is designed to account for the change in slope.
    This is done in a manner that resembles the Shirley method: 
    
    .. math::
        :label: slope
        
        \\frac{B_{\\text{Slope}}(E)}{dE} = -k_{\\text{Slope}} \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} ] \\, dE'

    where:

        - :math:`\\frac{B_{\\text{Slope}}(E)}{dE}` represents the slope of the background at energy :math:`E`,
        - :math:`I(E')` is the measured intensity at :math:`E'`,
        - :math:`I_{\\text{right}}` is the measured intensity of the rightmost datapoint,
        - :math:`k_{\\text{Slope}}` parameter to scale the integral to resemble the measured data. This parameter is related to the Tougaard background. For details see [8]_.
    
    To get the background itself, equation :math:numref:`slope` is integrated:
    
    .. math::
        :label: slope2

         B_{\\text{Slope}}(E)= \\int_{E}^{E_{\\text{right}}} [\\frac{B_{\\text{Slope}}(E')}{dE'}] \\, dE'

    
    .. table:: Model-specific available parameters
       :widths: auto

       +-----------+---------------+----------------------------------------------------------------------------------------+
       | Parameters|  Type         | Description                                                                            |
       +===========+===============+========================================================================================+
       | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
       +-----------+---------------+----------------------------------------------------------------------------------------+
       | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
       +-----------+---------------+----------------------------------------------------------------------------------------+
       | k         | :obj:`float`  | Slope parameter :math:`k_{\\text{Slope}}`.                                              |
       +-----------+---------------+----------------------------------------------------------------------------------------+
    
    Warning
    -------
    Please note that the Slope background should not be solely relied upon to mimic a measured XPS background. It is advisable to use it combined with other background models, such as the Shirley background.
    For further details, please refer to A. Herrera-Gomez et al [8]_. 
    
    Hint
    ----

    The `SlopeBG` class inherits from `lmfit.model.Model` and extends it with specific behavior and functionality related to the Slope background.

    Args
    ----
        `independent_vars` : :obj:`list` of :obj:`str`, optional
            Arguments to the model function that are independent variables
            default is ['x']).
        `prefix` : :obj:`str`, optional
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

    """+ lmfit.models.COMMON_INIT_DOC

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
