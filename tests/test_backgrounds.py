import numpy as np
import lmfit
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lmfitxps import backgrounds
import pytest

@pytest.fixture
def shirley_func():
    def create_shirley(*args, **kwargs):
        return backgrounds.shirley(*args, **kwargs)
    return create_shirley
def test_shirley_same_output(shirley_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    k = 0.003
    const = y[-1]

    result1 = shirley_func(y,k,const)
    result2 = shirley_func(y,k,const)

    assert np.array_equal(result1, result2)


@pytest.fixture
def shirley_calculate_func():
    def create_shirley_calculate(x,y,tol,maxit):
        return backgrounds.shirley_calculate(x, y, tol, maxit)

    return create_shirley_calculate
@pytest.mark.parametrize(
    "tol,maxit",
    [
        (5e-5,10),
        (0.5, 50),
    ],
)
def test_shirley_calculate_iterations(shirley_calculate_func, tol, maxit):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    expected= shirley_calculate_func(x=x, y=y, tol=1e-8, maxit=100)

    result = shirley_calculate_func(x=x, y=y, tol=tol, maxit=maxit)
    assert np.isclose(expected, result, rtol=1e-4).any()
    assert np.allclose(expected, result, rtol=1e-4)

@pytest.fixture()
def shirley_model():
    """Return a Shirley model."""
    return lmfit.Model(backgrounds.shirley, independent_vars=["y"])
def test_fit_shirley(shirley_model, shirley_calculate_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    params = lmfit.Parameters()
    y_shirley = shirley_calculate_func(x=x, y=y, tol=1e-8, maxit=100)
    params.add('k', value=0.0015)
    params.add('const', value=np.min(y))
    eva= shirley_model.eval(data=y, params=params, y=y)
    result = shirley_model.fit(y_shirley, params, y=y, weights=1/np.sqrt(y))
    assert result.success
    assert result.errorbars
    assert result.redchi/len(x) < 1000




@pytest.fixture
def tougaard_func():
    def create_tougaard(*args, **kwargs):
        return backgrounds.tougaard(*args, **kwargs)
    return create_tougaard
def test_tougaard_same_output(tougaard_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    B = 202.77
    C = 144.506
    C_d = 0.281
    D = 268.598
    extend = 0

    result1 = tougaard_func(x, y, B, C, C_d, D, extend)
    result2 = tougaard_func(x, y, B, C, C_d, D, extend)

    assert np.array_equal(result1, result2)


@pytest.fixture
def tougaard_calculate_func():
    def create_tougaard_calculate(x, y, tb, tc, tcd, td, maxit):
        return backgrounds.tougaard_calculate(x, y, tb, tc, tcd, td, maxit)

    return create_tougaard_calculate
@pytest.mark.parametrize(
    "B, C, C_d, D, maxit",
    [
        (212, 144.506, 0.281, 268.598, 10),
        (212, 144.506, 0.281, 268.598, 50),
    ],
)
def test_tougaard_calculate_iterations(tougaard_calculate_func, B, C, C_d, D, maxit):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    expected = tougaard_calculate_func(x=x, y=y, tb=212, tc=144.506, tcd=0.281, td=268.598, maxit=100)[0]
    result = tougaard_calculate_func(x=x, y=y, tb=B, tc=C, tcd=C_d, td=D, maxit=maxit)[0]
    assert np.isclose(expected, result, rtol=1e-4).any()
    assert np.allclose(expected, result, rtol=1e-4)

@pytest.fixture()
def tougaard_model():
    """Return a Tougaard model."""
    return lmfit.Model(backgrounds.tougaard, independent_vars=["y", 'x'])

def test_fit_tougaard(tougaard_calculate_func, tougaard_model):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    B = 202.77
    C = 144.506
    C_d = 0.281
    D = 268.598

    params = lmfit.Parameters()
    y_tougaard = tougaard_calculate_func(x=x, y=y, tb=B, tc=C, tcd=C_d, td=D, maxit=100)[0]
    params.add('B', value=200)
    params.add('C', value=144.506, vary=False)
    params.add('C_d', value=0.281, vary=False)
    params.add('D', value=268.598, vary=False)
    params.add('extend', value=50, vary=False)
    result = tougaard_model.fit(y_tougaard, params, y=y, x=x)
    assert result.success
    assert result.errorbars
    assert result.redchi / len(x) < 1000



@pytest.fixture
def slope_func():
    def create_slope(*args, **kwargs):
        return backgrounds.slope(*args, **kwargs)
    return create_slope
def test_slope_same_output(slope_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    y = data[:, 1]
    k=10

    result1 = slope_func(y,k)
    result2 = slope_func(y,k)

    assert np.array_equal(result1, result2)

@pytest.fixture()
def slope_model():
    """Return a Slope background model."""
    return lmfit.Model(backgrounds.slope, independent_vars=["y"])

def test_fit_slope(slope_model):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    y = data[:, 1]

    params = lmfit.Parameters()

    params.add('k', value=30, vary=False)
    result = slope_model.fit(y, params, y=y)
    y_slope=result.best_fit
    noise_std = np.sqrt(abs(y_slope))

    noise_mean = 0


    # Add the noise
    noisy_slope =y_slope+ np.random.normal(loc=noise_mean, scale=noise_std, size=len(y)) * np.random.choice([-1, 1], size=len(y))
    params = lmfit.Parameters()

    params.add('k', value=30, vary=True)
    result = slope_model.fit(noisy_slope, params, y=y)

    assert result.success
    assert result.errorbars
