import matplotlib.pyplot as plt
import numpy as np
import lmfit
from tests.context import backgrounds
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
        return backgrounds.shirley_calculate(x,y,tol,maxit)

    return create_shirley_calculate
@pytest.mark.parametrize(
    "tol,maxit, expected",
    [
        (5e-5,100,1897300),
        (0.5, 50,1897300),
    ],
)
def test_shirley_calculate_iterations(shirley_calculate_func, tol, maxit, expected):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]

    result = shirley_calculate_func(x=x, y=y, tol=tol, maxit=maxit)
    assert abs(np.sum(result)-expected)<100



def test_compare_shirleys(shirley_calculate_func, shirley_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    k = 0.00151
    const = y[-1]
    maxit = 100
    tol=1e-5

    result1 = shirley_calculate_func(x=x, y=y, tol=tol, maxit=maxit)
    result2 = shirley_func(y=y,k=k,const=const)
    assert abs(np.sum(result1-result2)/len(x))<50



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
    "B, C, C_d, D, maxit, expected",
    [
        (212, 144.506, 0.281, 268.598, 10,1572888),
        (212, 144.506, 0.281, 268.598, 50,1572888),
    ],
)
def test_tougaard_calculate_iterations(tougaard_calculate_func, B, C, C_d, D, maxit, expected):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]

    result = tougaard_calculate_func(x=x, y=y, tb=B, tc=C, tcd=C_d, td=D, maxit=maxit)
    assert abs(np.sum(result[0])-expected)<100000



def test_compare_tougaards(tougaard_calculate_func, tougaard_func):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    B = 202.77
    C = 144.506
    C_d = 0.281
    D = 268.598
    maxit = 100
    extend=50

    result1 = tougaard_calculate_func(x=x, y=y, tb=B, tc=C, tcd=C_d, td=D, maxit=maxit)
    result2 = tougaard_func(x=x, y=y, B=B, C=C, C_d=C_d, D=D, extend=extend)
    assert abs(np.sum(result1[0]-result2)/len(x))<200
