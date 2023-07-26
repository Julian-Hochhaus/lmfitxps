import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from scipy.signal import convolve as sc_convolve
__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "1.2.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def dublett(x, amplitude, sigma, gamma, gaussian_sigma, center, soc, height_ratio, fct_coster_kronig):
    """
    Calculates the convolution of a Doniach-Sunjic Dublett with a Gaussian. Thereby, the Gaussian acts as the
    convolution kernel.

    Parameters
    ----------
    x: array-like
        Array containing the energy of the spectrum to fit
    amplitude: float
        factor used to scale the calculated convolution to the measured spectrum. This factor is used as the amplitude
        of the Doniach profile.
    sigma: float
        Sigma of the Doniach profile
    gamma: float
        asymmetry factor gamma of the Doniach profile
    gaussian_sigma: float
        sigma of the gaussian profile which is used as the convolution kernel
    center: float
        position of the maximum of the measured spectrum
    soc: float
        distance of the second-highest peak (higher-bound-orbital) of the spectrum in relation to the maximum of the
        spectrum (the lower-bound orbital)
    height_ratio: float
        height ratio of the second-highest peak (higher-bound-orbital) of the spectrum in relation to the maximum of
        the spectrum (the lower-bound orbital)
    fct_coster_kronig: float
        ratio of the lorentzian-sigma of the second-highest peak (higher-bound-orbital) of the spectrum in relation to
        the maximum of the spectrum (the lower-bound orbital)
    Returns
    ---------
    array-type
        convolution of a doniach dublett and a gaussian profile
    """
    conv_temp = fft_convolve(
        doniach(x, amplitude=1, center=center, sigma=sigma, gamma=gamma) + doniach(x, height_ratio, center - soc,
                                                                                   fct_coster_kronig * sigma, gamma),
        1 / (np.sqrt(2 * np.pi) * gaussian_sigma) * gaussian(x, amplitude=1, center=np.mean(x), sigma=gaussian_sigma))
    return amplitude * conv_temp / max(conv_temp)


def singlett(x, amplitude, sigma, gamma, gaussian_sigma, center):
    """
    Calculates the convolution of a Doniach-Sunjic with a Gaussian.
    Thereby, the Gaussian acts as the convolution kernel.

    Parameters
    ----------
    x: array-like
        Array containing the energy of the spectrum to fit
    amplitude: float
        factor used to scale the calculated convolution to the measured spectrum. This factor is used as
        the amplitude of the Doniach profile.
    sigma: float
        Sigma of the Doniach profile
    gamma: float
        asymmetry factor gamma of the Doniach profile
    gaussian_sigma: float
        sigma of the gaussian profile which is used as the convolution kernel
    center: float
        position of the maximum of the measured spectrum

    Returns
    ---------
    array-type
        convolution of a doniach profile and a gaussian profile
    """
    conv_temp = fft_convolve(doniach(x, amplitude=1, center=center, sigma=sigma, gamma=gamma),
                             1 / (np.sqrt(2 * np.pi) * gaussian_sigma) * gaussian(x, amplitude=1, center=np.mean(x),
                                                                                  sigma=gaussian_sigma))
    return amplitude * conv_temp / max(conv_temp)


kb = 8.6173e-5  # Boltzmann k in eV/K


def fermi_edge(x, amplitude, center, kt, sigma):
    """
    Calculates the convolution of a Thermal Distribution (Fermi-Dirac Distribution) with a Gaussian.
    Thereby, the Gaussian acts as the convolution kernel.

    Parameters
    ----------
    x: array-like
        Array containing the energy of the spectrum to fit
    amplitude: float
        factor used to scale the calculated convolution to the measured spectrum. This factor is used
        as the amplitude of the Gaussian Kernel.
    center: float
        position of the step of the fermi edge
    kt: float
        boltzmann constant in eV multiplied with the temperature T in kelvin
        (i.e. for room temperature kt=kb*T=8.6173e-5 eV/K*300K=0.02585 eV)
    sigma: float
        Sigma of the gaussian profile which is used as the convolution kernel



    Returns
    ---------
    array-type
        convolution of a fermi dirac distribution and a gaussian profile
    """
    conv_temp = fft_convolve(thermal_distribution(x, amplitude=1, center=center, kt=kt, form='fermi'),
                             1 / (np.sqrt(2 * np.pi) * sigma) * gaussian(x, amplitude=1, center=np.mean(x),
                                                                         sigma=sigma))
    return amplitude * conv_temp / max(conv_temp)


def convolve(data, kernel):
    """
    Calculates the convolution of a data array with a kernel by using numpy convolve function.
    To suppress edge effects and generate a valid convolution on the full data range, the input dataset is extended
    at the edges.

    Parameters
    ----------
    data: array-like
        1D-array containing the data to convolve
    kernel: array-like
        1D-array which defines the kernel used for convolution

    Returns
    ---------
    array-type
        convolution of a data array with a kernel array

    See Also
    ---------
    numpy.convolve()
    """

    min_num_pts = min(len(data), len(kernel))
    padding = np.ones(min_num_pts)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    out = np.convolve(padded_data, kernel, mode='valid')
    n_start_data = int((len(out) - min_num_pts) / 2)
    return (out[n_start_data:])[:min_num_pts]


def fft_convolve(data, kernel):
    """
    Calculates the convolution of a data array with a kernel by using the convolution theorem and thereby
    transforming the time-consuming convolution operation into a multiplication of FFTs.
    For the FFT and inverse FFT, numpy implementation (which is basically the implementation used in scipy)
    of fft and ifft is used.
    To suppress edge effects and generate a valid convolution on the full data range, the input dataset is
    extended at the edges.

    Parameters
    ----------
    data: array-like
        1D-array containing the data to convolve
    kernel: array-like
        1D-array which defines the kernel used for convolution

    Returns
    ---------
    array-type
        convolution of a data array with a kernel array

    See Also
    ---------
    scipy.signal.convolve()
    """
    min_num_pts = min(len(data), len(kernel))
    padding = np.ones(min_num_pts)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    out = sc_convolve(padded_data, kernel, mode='valid', method="fft")
    n_start_data = int((len(out) - min_num_pts) / 2)
    return (out[n_start_data:])[:min_num_pts]

