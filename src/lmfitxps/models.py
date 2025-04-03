import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from .lineshapes import singlett, dublett, fermi_edge, convolve, fft_convolve
from .backgrounds import tougaard, slope, shirley
from lmfit import Model
import lmfit
from lmfit.models import guess_from_peak
from scipy.signal import convolve as sc_convolve
import scipy.constants

__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "4.1.2"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"


class ConvGaussianDoniachSinglett(lmfit.model.Model):
    __doc__ = ("""
    A model based on a convolution of a Gaussian and a Doniach-Sunjic profile. The model is designed for fitting XPS signals with asymmetry. 
    The Gaussian thereby represents the gaussian-like influences of the experimental setup and the Doniach-Sunjic represents the sample's physics.
    
    The implementation is based on the `Gaussian <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L46>`_ 
    and `Doniach <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L296>`_ 
    lineshapes of the LMFIT package. The convolution is calculated using the FFT-Convolution 
    implemented in `scipy.signal.fftconvolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_.
    
    The Convolution is, in analogy to the Voigt profile given by:
    
    .. math::
        
        (DS * G)(E, A, \\mu, \\gamma, \\alpha, \\sigma) = A \\cdot \\int_{-\\infty}^{\\infty} DS(E', \\mu,\\gamma,\\alpha) G(E - E', \\sigma)\\, dE'

    The Doniach-Sunjic profile (:math:`DS`) is convolved with the Gaussian kernel (:math:`G`).

    Thereby:
        - :math:`A` is the amplitude of the peak profile,
        - :math:`\\mu` is the center of the peak,
        - :math:`\\gamma` represents the broadening of the Doniach-Sunjic lineshape,
        - :math:`\\alpha` is the asymmetry of the Doniach-Sunjic lineshape,
        - :math:`\\sigma` is the broadening parameter of the Gaussian kernel.

    .. table:: Model-specific available parameters
        :widths: auto
        
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | Parameters     |  Type         | Description                                                                            |
        +================+===============+========================================================================================+
        | x              | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | y              | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | amplitude      | :obj:`float`  | amplitude :math:`A` of the peak profile                                                |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | sigma          | :obj:`float`  | Broadening of the Doniach-Sunjic.                                                      |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | gamma          | :obj:`float`  | Asymmetry of the Doniach-Sunjic.                                                       |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | center         | :obj:`float`  | Center of the peak profile.                                                            |
        +----------------+---------------+----------------------------------------------------------------------------------------+
        | gaussian_sigma | :obj:`float`  | Broadening of the gaussian kernel.                                                     |
        +----------------+---------------+----------------------------------------------------------------------------------------+


    Hint
    ----

    The `ConvGaussianDoniachSinglett` class inherits from `lmfit.model.Model` and only extends it. Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""

    """ + lmfit.models.COMMON_INIT_DOC)

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
        """
        Hint
        ----

        Needs improvement and does not work great yet. E.g. using peakfind.


        Generates initial parameter values for the model based on the provided data and optional arguments.

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        doniach_pars = guess_from_peak(Model(doniach), data, x, negative=False)
        gaussian_sigma = (doniach_pars["sigma"].value)
        params = self.make_params(amplitude=doniach_pars["amplitude"].value, sigma=doniach_pars["sigma"].value,
                                  gamma=doniach_pars["gamma"].value, gaussian_sigma=gaussian_sigma,
                                  center=doniach_pars["center"].value)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ConvGaussianDoniachDublett(lmfit.model.Model):
    __doc__ = ("""
    This model represents a dublett peak profile as observed in XPS spectra.
    The model is basically the sum of two singlett peak profiles separated by the energy distance of the two orbitals with different spin-orbit state.
    The implementation is therefore similar to the one of :ref:`ConvGaussianDoniachSinglett`.
    It based on a convolution of a Gaussian and the sum of two Doniach-Sunjic profiles.

    The implementation is based on the `Gaussian <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L46>`_ 
    and `Doniach <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L296>`_ 
    lineshapes of the LMFIT package. The convolution is calculated using the FFT-Convolution 
    implemented in `scipy.signal.fftconvolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_.

    .. math::

         &((DS_1+DS_2) * G)(E, A, \\mu, \\gamma, \\alpha, \\sigma, d, r, ckf) \\\\ 
         &= A \\cdot \\int_{-\\infty}^{\\infty} (DS_1(E', \\mu,\\gamma,\\alpha)+DS_2(E', \\mu,\\gamma,\\alpha, d,r,ckf)) G(E - E', \\sigma)\\, dE'

    Thereby:
        - :math:`A` is the amplitude of the larger peak of the dublett (P1),
        - :math:`r` defines the ratio of the height of the smaller dublett-peak (P2) with respect to the larger one,
        - :math:`\\mu` is the center of the larger peak (P1),
        - :math:`d` is the energy distance between the two peaks as an absolute value
        - :math:`\\gamma` represents the broadening of P1,
        - :math:`ckf` scales the broadening of P2 with respect to P1 to represent the Coster-Kronig effect.
        - :math:`\\alpha` is the asymmetry of both peaks,
        - :math:`\\sigma` is the broadening parameter of the Gaussian kernel.
        

    .. table:: Model-specific available parameters
        :widths: auto

        +------------------+---------------+----------------------------------------------------------------------------------------+
        | Parameters       |  Type         | Description                                                                            |
        +==================+===============+========================================================================================+
        | x                | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | y                | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | amplitude        | :obj:`float`  | amplitude :math:`A` of the larger peak (P1) of the dublett.                            |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | sigma            | :obj:`float`  | Doniach-broadening of the larger peak (P1) of the dublett.                             |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | gamma            | :obj:`float`  | Asymmetry of the larger peak (P1) of the dublett.                                      |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | center           | :obj:`float`  | Center of peak P1.                                                                     |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | soc              | :obj:`float`  | Distance (:math:`d`) in energy between the two peaks.                                  |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | height_ratio     | :obj:`float`  | Ratio (:math:`r`) of the amplitude of the smaller peak with respect to the larger one. |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | fct_coster_kronig| :obj:`float`  | Factor :math:`ckf` scales P2's broadening with respect to P1 (Coster-Kronig effect).   |
        +------------------+---------------+----------------------------------------------------------------------------------------+
        | gaussian_sigma   | :obj:`float`  | Broadening of the gaussian kernel.                                                     |
        +------------------+---------------+----------------------------------------------------------------------------------------+

    Hint
    ----

    The `ConvGaussianDoniachDublett` class inherits from `lmfit.model.Model` and only extends it. Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""

    """ + lmfit.models.COMMON_INIT_DOC)

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
        """
        Hint
        ----

        Needs improvement and does not work great yet. E.g. using peakfind.


        Generates initial parameter values for the model based on the provided data and optional arguments.

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
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
    __doc__ = ("""
        This Model function is intended to fit the Fermi edge in XPS spectra.
        To do so, a Fermi-Dirac Distribution is convoluted with a Gaussian.

        The implementation is based on the `Gaussian <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L46>`_ 
        and `thermal_distribution (form='fermi') <https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/lineshapes.py#L371>`_ 
        lineshapes of the LMFIT package. The convolution is calculated using the FFT-Convolution 
        implemented in `scipy.signal.fftconvolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_.

        The Convolution is given by:

        .. math::

           (FD * G)(E, A, \\mu, k_B T, \\sigma) = A \\cdot \\int_{-\\infty}^{\\infty} FD(E', \\mu,k_B T) G(E - E', \\sigma)\\, dE'


        Thereby is:
            - :math:`A` is the step-height of the Fermi edge,
            - :math:`\\mu` is the Fermi level,
            - :math:`k_B T` is the Boltzmann constant in eV/K multiplied by the temperature T in Kelvin (i.e. at room temperature: k_B T=k_B*T=8.6173e-5 eV/K*300K=0.02585 eV),
            - :math:`\\sigma` is the broadening parameter of the Gaussian kernel.

        .. table:: Model-specific available parameters
            :widths: auto

            +----------------+---------------+----------------------------------------------------------------------------------------+
            | Parameters     |  Type         | Description                                                                            |
            +================+===============+========================================================================================+
            | x              | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
            +----------------+---------------+----------------------------------------------------------------------------------------+
            | y              | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
            +----------------+---------------+----------------------------------------------------------------------------------------+
            | amplitude      | :obj:`float`  | step height :math:`A` of the fermi edge.                                               |
            +----------------+---------------+----------------------------------------------------------------------------------------+
            | center         | :obj:`float`  | position :math:`\mu` of the edge (Fermi level)                                         |
            +----------------+---------------+----------------------------------------------------------------------------------------+
            | kt             | :obj:`float`  | Boltzmann constant in eV/K multiplied by temperature T in Kelvin (:math:`k_B T`)       |
            +----------------+---------------+----------------------------------------------------------------------------------------+
            | sigma          | :obj:`float`  | Broadening of the gaussian kernel.                                                     |
            +----------------+---------------+----------------------------------------------------------------------------------------+


    Hint
    ----

    The `FermiEdgeModel` class inherits from `lmfit.model.Model` and only extends it. Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""

    """ + lmfit.models.COMMON_INIT_DOC)

    def __init__(self, *args, **kwargs):
        super().__init__(fermi_edge, *args, **kwargs)
        # limit several input parameters to positive values
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('kt', value=0.02585, min=0)  # initial value is room temperature
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('center', value=100)
        self.set_param_hint('amplitude', value=100, min=0)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        kb = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.set_param_hint('center', value=np.mean(x), min=min(x), max=max(x))
        self.set_param_hint('kt', value=kb * 300, min=0, max=kb * 1500)
        self.set_param_hint('amplitude', value=(max(data) - min(data)) / 10, min=0, max=(max(data) - min(data)))
        self.set_param_hint('sigma', value=(max(x) - min(x)) / len(x), min=0, max=2)
        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class TougaardBG(lmfit.model.Model):
    __doc__ = ("""
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

    The `TougaardBG` class inherits from `lmfit.model.Model` and extends it with specific behavior and functionality related to the Tougaard 4-parameter loss function. Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""
    
    
    """ + lmfit.models.COMMON_INIT_DOC)

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
        self.set_param_hint('B', value=2886)
        self.set_param_hint('C', value=1643)
        self.set_param_hint('C_d', value=1)
        self.set_param_hint('D', value=1)
        self.set_param_hint('extend', value=0, vary=False)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

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
    __doc__ = ("""
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
    
    The `ShirleyBG` class inherits from `lmfit.model.Model` and acts as a predefined model for calculating the Shirley background.Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""
    
    """ + lmfit.models.COMMON_INIT_DOC)

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
        self.set_param_hint('k', value=0.03)
        self.set_param_hint('const', value=1000)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

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
    __doc__ = ("""
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

    The `SlopeBG` class inherits from `lmfit.model.Model` and extends it with specific behavior and functionality related to the Slope background. Therefore, the `lmfit.model.Model` class parameters are inherited as well.


    **LMFIT: Common models documentation**
    """"""""""""""""""""""""""""""""""""
        
    """+ lmfit.models.COMMON_INIT_DOC)

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

        :param data: Array containing the data (=intensities) to fit.
        :type data: array-like
        :param x: Array containing the independent variable (=energy).
        :type x: array-like
        :param kwargs: Initial guesses for the parameters of the model function.

        :returns: Initial parameter values for the model.
        :rtype: lmfit.Parameters


        :note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(k=0.01)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
