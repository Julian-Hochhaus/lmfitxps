import numpy as np
import lmfit
from src.lmfitxps import models
import pytest

@pytest.fixture()
def dublett_model():
    """Return a Doniach-Sunjic convolved Dublett peak model."""
    return models.ConvGaussianDoniachDublett(prefix='peak_')
@pytest.fixture()
def tougaard_model():
    """Return a Tougaard background model."""
    return models.TougaardBG(prefix='tougaard_', independent_vars=['x', 'y'])

def test_fit_tougaard_dublett(tougaard_model, dublett_model):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]

    params = lmfit.Parameters()

    params.add('tougaard_B', value=200)
    params.add('tougaard_C', value=144.506, vary=False)
    params.add('tougaard_C_d', value=0.281, vary=False)
    params.add('tougaard_D', value=268.598, vary=False)
    params.add('tougaard_extend', value=50, vary=False)
    params.add('peak_amplitude', value=80000, min=0)
    params.add('peak_sigma', value=0.2, min=0)
    params.add('peak_gamma', value=0.02)
    params.add('peak_gaussian_sigma', value=0.2, min=0)
    params.add('peak_center', value=92)
    params.add('peak_soc', value=3.67, vary=False)
    params.add('peak_height_ratio', value=0.75, min=0)
    params.add('peak_fct_coster_kronig', value=1, min=0)
    fit_model=tougaard_model+dublett_model
    result = fit_model.fit(y, params, y=y, x=x)
    assert result.success
    assert result.errorbars
