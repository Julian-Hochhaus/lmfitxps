import numpy as np
import copy
__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "2.4.2"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def tougaard_closure():
    bgrnd = [[], [], []]  # This will act as the closure to store the precalculated data
    def tougaard_helper(x, y, B, C, C_d, D, extend=0):
        """
        .. Hint::
            This function employs a closure to calculate the Tougaard sum only once and subsequently accesses it during subsequent executions of the optimization procedure. When calling `tougaard()`, you are effectively accessing the inner, nested function.

            The concept is as follows:

        .. code-block:: python

            def tougaard_closure():
                bgrnd=[] #store the Tougaard background to optimize performance by avoiding recalculations
                def tougaard_helper():
                    # do actual calculation

            tougaard = tougaard_closure()

        The Tougaard backlground is based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [1]_.

        | In addition to R.Hesse's approach, this model introduces the `extend` parameter, for details, please refer to :ref:`extend_parameter`.

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

        For further details on the 2-PIESCS loss function, please refer to S.Tougaard [2]_, and for the 3-PIESCS loss function, see S. Tougaard [3]_.


        .. table::
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
            | extend    | :obj:`float`  | Determines, how far the spectrum is extended on the right (in eV). Defaults to 0.      |
            +-----------+---------------+----------------------------------------------------------------------------------------+

        Note
        ----
        This function is used as the model function in the :ref:`TougaardBG` lmfitxps model.
        """

        nonlocal bgrnd

        if np.array_equal(bgrnd[0], y) and bgrnd[2][0] == extend:
            # Check if loss function was already calculated
            return [B * elem for elem in bgrnd[1]]
        else:
            bgrnd[0] = copy.deepcopy(y)
            bgrnd[2] = [extend]
            bg = []
            delta_x = (x[-1] - x[0]) / len(x)
            len_padded = abs(int(extend / delta_x))
            padded_x = np.concatenate((x, np.linspace(x[-1] + delta_x, x[-1] + delta_x * len_padded, len_padded)))
            padded_y = np.concatenate((y, np.mean(y[-10:]) * np.ones(len_padded)))
            for k in range(len(x)):
                x_k = x[k]
                bg_temp = 0
                for j in range(len(padded_y[k:])):
                    padded_x_kj = padded_x[k + j]
                    bg_temp += abs(padded_x_kj - x_k) / ((C + C_d * (padded_x_kj - x_k) ** 2) ** 2
                                                      + D * (padded_x_kj - x_k) ** 2) * padded_y[k + j] * abs(delta_x)
                bg.append(bg_temp)
            bgrnd[1] = bg
            return np.asarray([B * elem for elem in bgrnd[1]])
    return tougaard_helper

# Create the tougaard function with the closure
tougaard = tougaard_closure()

def shirley(y, k, const):
    """
    Calculates the Shirley background for X-ray photoelectron spectroscopy (XPS) spectra by integrating the step characteristic of the spectrum.
    For further details, please refer to Shirley [5]_ or Jansson et al. [6]_.

    Hint
    ----
    The Shirley background is typically calculated iteratively using the following formula:

    .. math::

        B_{S, n}(E) = k_n \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} - B_{S, n-1}(E')] \\, dE'

    Using this iterative process, makes it necessary to calculate the Shirley background before the fitting procedure, which is not always meaningful.
    If you want to use this approach, please use the :ref:`shirley_calculate` function.


    Here the Shirley background is computed according to:

    .. math::

        B_S(E)=k\\cdot \\int_{E}^{E_{\\text{right}}}\\left[I(E')-I_{\\text{right}}\\right] \\, dE'

    In the actual implementation of the function, :math:`I_{\\text{right}}` corresponds to `const` and is substracted from the intensity data y (:math:`I(E)`) before calculating the integral by summing over the intensities.
    This approach allows to include the Shirley background into the fitting model (e.g. as implemented in the :ref:`ShirleyBG` lmfitxps model) and to be adaptively determined during the fitting process, while still preserving the iterative concept of the Shirley's background calculation.

    .. table::
        :widths: auto

        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | Parameters | Type          | Description                                                                                        |
        +============+===============+====================================================================================================+
        | y          | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                                    |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | k          | :obj:`float`  | Shirley parameter :math:`k`, determines step-height of the Shirley background.                     |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
        | const      | :obj:`float`  | Constant value added to the step-like Shirley background, often set to :math:`I_{\\text{right}}`.   |
        +------------+---------------+----------------------------------------------------------------------------------------------------+
    Note
    ----
    This function is used as the model function in the :ref:`ShirleyBG` lmfitxps model.

    """
    n = len(y)
    y_right = const
    y_temp = y - y_right  # step characteristic is better approximated if only the step without  constant background offset is integrated
    bg = []
    for i in range(n):
        bg.append(np.sum(y_temp[i:]))
    return np.asarray([k * elem + y_right for elem in bg])

def slope(y, k):
    """
    Calculates the Slope background for X-ray photoelectron spectroscopy (XPS) spectra.
    The Slope Background is implemented as suggested by A. Herrera-Gomez et al in [4]_.
    Hereby, while the Shirley background is designed to account for the difference in background height between the two sides of a peak, the Slope background is designed to account for the change in slope.
    This is done in a manner that resembles the Shirley method:

    .. math::

        \\frac{B_{\\text{Slope}}(E)}{dE} = -k_{\\text{Slope}} \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} ] \\, dE'

    where:

        - :math:`\\frac{B_{\\text{Slope}}(E)}{dE}` represents the slope of the background at energy :math:`E`,
        - :math:`I(E')` is the measured intensity at :math:`E'`,
        - :math:`I_{\\text{right}}` is the measured intensity of the rightmost datapoint,
        - :math:`k_{\\text{Slope}}` parameter to scale the integral to resemble the measured data. This parameter is related to the Tougaard background. For details see [4]_.

    To get the background itself, equation :math:numref:`slope` is integrated:

    .. math::

         B_{\\text{Slope}}(E)= \\int_{E}^{E_{\\text{right}}} [\\frac{B_{\\text{Slope}}(E')}{dE'}] \\, dE'


    .. table::
       :widths: auto

       +-----------+---------------+----------------------------------------------------------------------------------------+
       | Parameters|  Type         | Description                                                                            |
       +===========+===============+========================================================================================+
       | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
       +-----------+---------------+----------------------------------------------------------------------------------------+
       | k         | :obj:`float`  | Slope parameter :math:`k_{\\text{Slope}}`.                                              |
       +-----------+---------------+----------------------------------------------------------------------------------------+

    Note
    ----
    This function is used as the model function in the :ref:`SlopeBG` lmfitxps model

    Warning
    -------
    Please note that the Slope background should not be solely relied upon to mimic a measured XPS background. It is advisable to use it combined with other background functions, such as the Shirley background.
    For further details, please refer to A. Herrera-Gomez et al [4]_.
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
    Calculates the Shirley background for a given set of x (energy) and y (intensity) data.

    The implementation was inspired by the python implementation of Kane O'Donnell [5]_.

    The Shirley background is calculated iteratively:

    .. math::
        :label: shirleystatic

        B_{S, n}(E) = k_n \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} - B_{S, n-1}(E')] \\, dE'


    where:
        - :math:`B_{S, n}(E)` represents the Shirley background at :math:`E` in the :math:`n`-th iteration,
        - :math:`I(E')` is the intensity at :math:`E'`,
        - :math:`k_n` is the Shirley scaling parameter for the :math:`n`-th iteration.
        - :math:`E_{\\text{right}}` and :math:`I_{\\text{right}}` are the rightmost energy/intensity of the dataset.

    The iterative process continues until the difference :math:`B_{S, n}(E) - B_{S, n-1}(E)` is suitable small or the number of maximum iterations :math:`maxit` is exceeded.

    Initially, :math:`B_{S, 0}(E)=0` is choosen and :math:`k_n` is found from the requirement, that :math:`\\left(I_{\\text{left}}-B_{S, n}(E_{\\text{left}})\\right)=0`.
    For further details, please refer to e.g. S. Tougaard [6]_ .

    Typically, convergence is reached after :math:`\\approx 5` iterations. The convergence criterion is:

     .. math::
        :label: shirleyconvergence

        \\left(B_{S, n}(E)-B_{S, n-1}(E)\\right)^2<tol


    Parameters:
    -----------

    .. table:: Available parameters
        :widths: auto

        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+
        | Parameter |  Type         | Description                                                                                                                    |
        +===========+===============+================================================================================================================================+
        | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                                                                   |
        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+
        | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                                                                |
        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+
        | tol       | :obj:`float`  | Tolerance used to determine, when the convergence is reached in equation :math:numref:`shirleyconvergence`. Defaults to 1e-5.  |
        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+
        | maxit     | :obj:`int`    | Maximum number of iterations before calculation is interrupted. Defaults to 10.                                                |
        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+

    Returns:
    --------
        :obj:`array`:  The function returns the calculated Shirley background as an :obj:`array`.

    Hint
    ----

    This function should be used, if you intend to calculate and remove the background from your data before starting the fitting procedure, if you instead wish to include the background in the fitting model, please use the desired background model, e.g. :ref:`ShirleyBG`.

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
        x = copy.deepcopy(x[::-1])
        y = copy.deepcopy(y[::-1])
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

    The implementation was inspired by the IGOR implementation of James Mudd [2]_.

    The Tougaard background is calculated using:

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

    During the calculation, the Tougaard background is calculated using the provided start parameters based on equation :math:numref:`tougaard`. This process is iteratively repeated, adapting the :math:`B` parameter, until convergence is reached or the number of iterations exceeds :math:`maxit`.

    The convergence is hereby defined by the deviation between the calculated Tougaard background :math:`B_T(E)` and the measured intensity :math:`y(E)` at the leftmost datapoint.

    The Tougaard background is considered to converge if :math:`|B_T(E)-y(E)|< 10^{-6}\\cdot B_T(E)` is fulfilled.

    Parameters:
    -----------

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
    --------
        :obj:`tuple` of (:obj:`numpy.ndarray`, :obj:`float`):  The function returns a tuple consisting of the calculated Tougaard background as a :obj:`numpy.array` and the Tougaard scale parameter :math:`B` as :obj:`float`.

    Hint
    ----

    This function should be used, if you intend to calculate and remove the background from your data before starting the fitting procedure, if you instead wish to include the background in the fitting model, please use the desired background model, e.g. :ref:`TougaardBG`.

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
