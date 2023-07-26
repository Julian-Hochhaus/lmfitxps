import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from lineshapes import singlett, dublett, fermi_edge, convolve, fft_convolve
from backgrounds import tougaard, slope, shirley
from lmfit import Model
from lmfit.models import guess_from_peak
from scipy.signal import convolve as sc_convolve

__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "1.2.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"


class ConvGaussianDoniachSinglett(lmfit.model.Model):
    __doc__ = "Model of a Doniach dublett profile convoluted with a gaussian. " \
              "See also lmfit->lineshape.gaussian and lmfit->lineshape.doniach." + lmfit.models.COMMON_INIT_DOC

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
    __doc__ = "Model of a Doniach profile convoluted with a gaussian. " \
              "See also lmfit->lineshape.gaussian and lmfit->lineshape.doniach." + lmfit.models.COMMON_INIT_DOC

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
    __doc__ = "Model of a ThermalDistribution convoluted with a gaussian. " \
              "See also lmfit->lineshape.gaussian and lmfit->lineshape.thermal_distribution." \
              + lmfit.models.COMMON_INIT_DOC

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
    Model of the 4 parameter loss function Tougaard.

    The implementation is based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [
    https://doi.org/10.1002/sia.3746]. In addition, the extend parameter is introduced, which improves the agreement
    between data and Tougaard BG by extending the data on the high-kinetic energy side (low binding energy side) by
    the mean intensity value at the rightmost kinetic energy scale. extend represents the length of the data
    extension on the high-kinetic-energy side in eV. Defaults to 30.

    Attributes:
        All attributes are inherited from the lmfit.model.Model class.

    Methods:
        __init__(*args, **kwargs):
            Initializes the TougaardBG model instance. Calls the super().__init__() method of the parent class
            (lmfit.model.Model) and sets parameter hints using _set_paramhints_prefix() method.

        _set_paramhints_prefix():
            Sets parameter hints for the model. Sets initial values and constraints for the parameters 'B', 'C',
            'C_d', 'D', and 'extend'.

        guess(data, x=None, **kwargs):
            Generates initial parameter values for the model based on the provided data and optional arguments.

    Note:
        The TougaardBG class inherits from lmfit.model.Model and extends it with specific behavior and functionality
        related to the Tougaard 4 parameter loss function.
    """ + lmfit.models.COMMON_INIT_DOC

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
        self.set_param_hint('extend', value=30, vary=False)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        Parameters:
            data (array-like): Array containing the data (=intensities) to fit.
            x (array-like): Array containing the independent variable values.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Parameters: Initial parameter values for the model.

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
        __init__(*args, **kwargs):
            Initializes the ShirleyBG model instance. Calls the super().__init__() method of the parent class
            (lmfit.model.Model) and sets parameter hints using _set_paramhints_prefix() method.

        _set_paramhints_prefix():
            Sets parameter hints for the model. Sets initial values and constraints for the parameters 'k' and 'const'.

        guess(data, x=None, **kwargs):
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

        The method sets initial values and constraints for the parameters 'k' and 'const'.

        """
        self.set_param_hint('k', value=0.03, min=0)
        self.set_param_hint('const', value=1000, min=0)

    def guess(self, data, x=None, **kwargs):
        """
        Generates initial parameter values for the model based on the provided data and optional arguments.

        Parameters:
            data (array-like): Array containing the data to fit.
            x (array-like): Array containing the independent variable values.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Parameters: Initial parameter values for the model.

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
        __init__(*args, **kwargs):
            Initializes the SlopeBG model instance. Calls the super().__init__() method of the parent class
            (lmfit.model.Model) and sets parameter hints using _set_paramhints_prefix() method.

        _set_paramhints_prefix():
            Sets parameter hints for the model. Sets an initial value for the parameter 'k'.

        guess(data, x=None, **kwargs):
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

        Parameters:
            data (array-like): Array containing the data to fit.
            x (array-like): Array containing the independent variable values.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Parameters: Initial parameter values for the model.

        Note:
            The method requires the 'x' parameter to be provided.
        """
        if x is None:
            return
        params = self.make_params(k=0.01)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
