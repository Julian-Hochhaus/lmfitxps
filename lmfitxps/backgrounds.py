import numpy as np
__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "1.3.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def tougaard_closure():
    """
    Calculates the Tougaard background of an X-ray photoelectron spectroscopy (XPS) spectrum.
    The following implementation is based on the four-parameter loss function (4-PIESCS)
    as suggested by [R.Hesse](https://doi.org/10.1002/sia.3746). In contrast to R.Hesse, the Tougaard background is not leveled with the data
    using a constant, but the background on the high-energy side is extended. This approach was found to lead to
    great convergence empirically, however, the length of the data extension remains arbitrary.
    To reduce computing time, as long as only B should be variated (which makes sense in most cases), if the loss
    function was already calculated, only B is further optimized.
    The 2-PIESCS loss function is created by using C_d=1 and D=0. Using C_d=-1 and D!=0 leads to the 3-PIESCS loss
    function.
    For further details on the 2-PIESCS loss function, see https://doi.org/10.1016/0038-1098(87)90166-9, and for the
    3-PIESCS loss function, see https://doi.org/10.1002/(SICI)1096-9918(199703)25:3<137::AID-SIA230>3.0.CO;2-L

    Parameters
    ----------
    x : array-like
        1D-array containing the x-values (energies) of the spectrum.
    y : array-like
        1D-array containing the y-values (intensities) of the spectrum.
    B : float
        B parameter of the 4-PIESCS loss function as introduced by R.Hesse (https://doi.org/10.1002/sia.3746).
        Acts as scaling factor of the Tougaard background model.
    C : float
        C parameter of the 4-PIESCS loss function as introduced by R.Hesse (https://doi.org/10.1002/sia.3746).
    C_d : float
        C' parameter of the 4-PIESCS loss function as introduced by R.Hesse (https://doi.org/10.1002/sia.3746).
        Set to 1 for the 2-PIESCS loss function. (and D to 0). Set to -1 for the 3-PIESCS loss function (D!=0).
    D : float
        D parameter of the 4-PIESCS loss function as introduced by R.Hesse (https://doi.org/10.1002/sia.3746).
        Set to 0 for the 2-PIESCS loss function (and C_d to 1). Set to !=0 for the 3-PIESCS loss function (C_d=-1).
    extend : float, optional
        Length of the data extension on the high-kinetic-energy side. Defaults to 0.
    Returns
    -------
    array-like
        The Tougaard background of the XPS spectrum.
    See Also
    --------
    The following implementation is based on the four-parameter loss function as suggested by
    R.Hesse [https://doi.org/10.1002/sia.3746].
    """
    bgrnd = [[], [], []]  # This will act as the closure to store the precalculated data
    def tougaard_helper(x, y, B, C, C_d, D, extend=0):

        nonlocal bgrnd

        if np.array_equal(bgrnd[0], y) and bgrnd[2][0] == extend:
            # Check if loss function was already calculated
            return [B * elem for elem in bgrnd[1]]
        else:
            bgrnd[0] = y
            bgrnd[2] = [extend]
            bg = []
            delta_x = abs((x[-1] - x[0]) / len(x))
            len_padded = int(extend / delta_x)
            padded_x = np.concatenate((x, np.linspace(x[-1] + delta_x, x[-1] + delta_x * len_padded, len_padded)))
            padded_y = np.concatenate((y, np.mean(y[-10:]) * np.ones(len_padded)))
            for k in range(len(x)):
                x_k = x[k]
                bg_temp = 0
                for j in range(len(padded_y[k:])):
                    padded_x_kj = padded_x[k + j]
                    bg_temp += (padded_x_kj - x_k) / ((C + C_d * (padded_x_kj - x_k) ** 2) ** 2
                                                      + D * (padded_x_kj - x_k) ** 2) * padded_y[k + j] * delta_x
                bg.append(bg_temp)
            bgrnd[1] = bg
            return np.asarray([B * elem for elem in bgrnd[1]])
    return tougaard_helper

# Create the tougaard function with the closure
tougaard = tougaard_closure()

def shirley(y, k, const):
    """
    Calculates the Shirley background of an X-ray photoelectron spectroscopy (XPS) spectrum.
    This implementation calculates the Shirley background by integrating the step characteristic of the spectrum.

    Parameters
    ----------
    y : array-like
        1D-array containing the y-values (intensities) of the spectrum.
    k : float
        Slope of the step characteristic.
    const : float
        Constant offset of the step characteristic.

    Returns
    -------
    array-like
        The Shirley background of the XPS spectrum.
    """
    n = len(y)
    y_right = const
    y_temp = y - y_right  # step characteristic is better approximated if only the step without background is integrated
    bg = []
    for i in range(n):
        bg.append(np.sum(y_temp[i:]))
    return np.asarray([k * elem + y_right for elem in bg])

def slope(y, k):
    """
    Calculates the slope background of an X-ray photoelectron spectroscopy (XPS) spectrum.
    The slope background has some similarities to the Shirley background, e.g. the slope background is calculated
    by integrating the Shirley background from each data point to the end.
    Afterwards, a slope parameter k is used to scale the slope accordingly to the measured data.

    Parameters
    ----------
    y : array-like
        1D-array containing the y-values (intensities) of the spectrum.
    k : float
        Slope of the linear function for determining the background.

    Returns
    -------
    array-like
        The slope background of the XPS spectrum.

    See Also
    --------
    Slope Background implemented as suggested by A. Herrera-Gomez et al in [DOI: 10.1016/j.elspec.2013.07.006].
    """
    n = len(y)
    y_right = np.min(y)
    y_temp = y - y_right
    temp = []
    bg = []
    for i in range(n):
        temp.append(np.sum(y_temp[i:]))
    for j in range(n):
        bg.append(np.sum(temp[j:]))
    return np.asarray([-k * elem for elem in bg])


def shirley_calculate(x, y, tol=1e-5, maxit=10):
    """
    Calculate the Shirley background for a given set of x and y data.
    Originally inspired by https://github.com/kaneod/physics/blob/master/python/specs.py.

    Args:
        x (array-like): The x-values of the data.
        y (array-like): The y-values of the data.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-5.
        maxit (int, optional): Maximum number of iterations. Defaults to 10.

    Returns:
        array: The Shirley background calculated from the input data.
    """

    n = len(y)
    # Sanity check: Do we actually have data to process here?
    # print(any(x), any(y), (any(x) and any(y)))
    if not (any(x) and any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return x * 0

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    yl = y[0]
    yr = y[-1]

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = y * 0

    Bnew = B.copy()

    it = 0
    while it < maxit:
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(n - 1):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1] - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(n):
            ysum = 0.0
            for j in range(i, n - 1):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] + y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        # if norm(Bnew - B) < tol:
        B = Bnew - B
        # print(it, (B**2).sum(), tol**2)
        if (B ** 2).sum() < tol ** 2:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("Max iterations exceeded before convergence.")
    if is_reversed:
        # print("Shirley BG: tol (ini = ", tol, ") , iteration (max = ", maxit, "): ", it)
        return (yr + B)[::-1]
    else:
        # print("Shirley BG: tol (ini = ", tol, ") , iteration (max = ", maxit, "): ", it)
        return yr + B


def tougaard_calculate(x, y, tb=2866, tc=1643, tcd=1, td=1, maxit=100):
    """
    Calculates the Tougaard background for a given set of x and y data. 
    The calculation is hereby based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [1]_.
    The implementation was inspired by the IGOR implementation [2]_.

    The Tougaard background is thereby calculated using:
    .. math::
        :label: tougaard

        B_T(E) = \\int_{E}^{\\infty} \\frac{B \\cdot T}{{(C + C_d \\cdot T^2)^2} + D \\cdot T^2} \\cdot y(E') \\, dE'

    where:

        - :math:`B_T(E)` represents the Tougaard background at energy :math:`E`,
        - :math:`y(E')` is the measured intensity at :math:`E'`,
        - :math:`T` is the energy difference :math:`E' - E`.
        - :math:`B` parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. Acts as the scaling factor for the Tougaard background model. This parameter is the only one optimized/variated during the calculation.
        - :math:`C` , :math:`C_d` and :math:`D` are parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. These parameters are kept fixed during the calculation.

    To generate the 2-PIESCS loss function, set :math:`C_d` to 1 and :math:`D` to 0.
    Set :math:`C_d=1` and :math:`D !=`  :math:`0` to get the 3-PIESCS loss function.

    For further details on the 2-PIESCS loss function, please refer to S.Tougaard [3]_, and for the
    3-PIESCS loss function, see S. Tougaard [4]_.

    During the calculation, the Tougaard background is calculated using the provided start parameters based on equation :math:numref:`tougaard`. This process is repeated, adapting the :math:`B` until convergence is reached or the number of iterations exceeds :math:`maxit`.
    The convergence is hereby defined by the deviation between the calculated Tougaard background :math:`B_T(E)` and the measured intensity :math:`y(E)` at the leftmost datapoint.
    The background is considered to converge if :math:`|B_T(E)-y(E)|< 10^{-6}\\cdot B_T(E)` is fulfilled.

    .. table:: Available parameters
        :widths: auto

        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | Parameter |  Type         | Description                                                                                                                                    |
        +===========+===============+================================================================================================================================================+
        | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                                                                                   |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                                                                                |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tb        | :obj:`float`  | B parameter of the 4-PIESCS loss function [1]_. Acts as scaling parameter and is optimized during the fit. Defaults to 2866 as starting value. |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tc        | :obj:`float`  | C parameter of the 4-PIESCS loss function [1]_. Defaults to 1643.                                                                              |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tcd       | :obj:`float`  | C' parameter of the 4-PIESCS loss function [1]_. Defaults to 1.                                                                                |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | td        | :obj:`float`  | D parameter of the 4-PIESCS loss function [1]_. Defaults to 1.                                                                                 |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | maxit     | :obj:`int`    | Maximum number of iterations before calculation is interrupted. Defaults to 100.                                                               |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+

    Returns:
        :obj:`tuple` of (:obj:`numpy.ndarray`, :obj:`float`):  The function returns a tuple consisting of the calculated Tougaard background as a :obj:`numpy.array` and the Tougaard scale parameter :math:`B` as :obj:`float`.
    """
    # Sanity check: Do we actually have data to process here?
    if not (np.any(x) and np.any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return [np.asarray(x * 0), tb]

    # KE in XPS or PE in XAS
    if x[0] < x[-1]:
        is_reversed = True
    # BE in XPS
    else:
        is_reversed = False

    Btou = y * 0

    it = 0
    while it < maxit:
        if not is_reversed:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (y[j] - y[len(y) - 1]) * (x[0] - x[1]) * (x[i] - x[j]) / (
                            (tc + tcd * (x[i] - x[j]) ** 2) ** 2 + td * (x[i] - x[j]) ** 2)
                Btou[i] = Bint * tb

        else:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (y[j] - y[len(y) - 1]) * (x[1] - x[0]) * (x[j] - x[i]) / (
                            (tc + tcd * (x[j] - x[i]) ** 2) ** 2 + td * (x[j] - x[i]) ** 2)
                Btou[i] = Bint * tb

        Boffset = Btou[0] - (y[0] - y[len(y) - 1])
        if abs(Boffset) < (0.000001 * Btou[0]) or maxit == 1:
            break
        else:
            tb = tb - (Boffset / Btou[0]) * tb * 0.5
        it += 1

    print("Tougaard B:", tb, ", C:", tc, ", C':", tcd, ", D:", td)

    return np.asarray(y[len(y) - 1] + Btou), tb
