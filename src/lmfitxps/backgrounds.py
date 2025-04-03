import numpy as np
import copy
from scipy.integrate import cumulative_trapezoid
__author__ = "Julian Andreas Hochhaus, Florian Kraushofer"
__copyright__ = "Copyright 2025"
__credits__ = ["Julian Andreas Hochhaus", "Florian Kraushofer"]
__license__ = "MIT"
__version__ = "4.1.2"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def tougaard_closure():
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
    bgrnd = [None, None, None] # This will act as the closure to store the precalculated data

    def tougaard_helper(x, y, B, C, C_d, D, extend=0):
        nonlocal bgrnd
        extend = int(extend)
        if bgrnd[0] is not None and np.array_equal(bgrnd[0], y) and int(bgrnd[2]) == extend:
            return B * np.array(bgrnd[1])
        bgrnd[0] = np.copy(y)
        bgrnd[2] = extend

        delta_x = abs((x[-1] - x[0])) / len(x)
        len_padded = abs(int(extend / delta_x))
        padded_x = np.concatenate([x, np.linspace(x[-1] + delta_x, x[-1] + delta_x * len_padded, len_padded)])
        padded_y = np.concatenate([y, np.full(len_padded, np.mean(y[-10:]))])

        bg = np.zeros_like(x)
        for k, x_k in enumerate(x):
            dx = padded_x[k:] - x_k
            denominator = (C + C_d * dx ** 2) ** 2 + D * dx ** 2
            bg[k] = np.sum(np.abs(dx) / denominator * padded_y[k:] * delta_x)
        bgrnd[1] = bg
        return B * bg

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
    y_right = const
    bg = np.cumsum(y)[::-1]
    return k * bg + y_right

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

    y_right = np.min(y)
    y_temp = y - y_right
    temp = np.cumsum(y_temp[::-1])[::-1]
    bg = np.cumsum(temp[::-1])[::-1]

    return -k * bg


def shirley_calculate(x, y, tol=1e-5, maxit=10, bounds=None):
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

        \\langle\\left(B_{S, n}(E)-B_{S, n-1}(E)\\right)^2\\rangle<tol


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
        | bounds    | :obj:`tuple`  | Either two x values or two (x,y) pairs. Determines the edges of the Shirley background. Background will be constant outside this range. If only x is passed, picks the y of closest data point. If nothing is passed, uses the edges of the data range.    |
        +-----------+---------------+--------------------------------------------------------------------------------------------------------------------------------+

    Returns:
    --------
        :obj:`array`:  The function returns the calculated Shirley background as an :obj:`array`.

    Hint
    ----

    This function should be used, if you intend to calculate and remove the background from your data before starting the fitting procedure, if you instead wish to include the background in the fitting model, please use the desired background model, e.g. :ref:`ShirleyBG`.

    """

    # Sanity check: Do we actually have data to process here?
    if not (any(x) and any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return x * 0            # TODO: raise ValueError instead?
    if not len(x) == len(y):
        print("Length missmatch between x and y. Returning zero background")
        return x * 0            # TODO: raise ValueError instead?

    # couple x and y values for easier handling in the following;
    #  data will be modified in-place, but this keeps the input x,y safe.
    data = np.array((x, y))

    if not bounds:
        bounds = np.array((data[:, 0], data[:, -1])).T
    else:
        bounds = np.array(bounds).T
        if bounds.shape == (2,):
            # bounds are only energies, don't have values yet.
            # cut the range, then use closest data values.
            data = data[:, (data[0] >= np.min(bounds)) &
                           (data[0] <= np.max(bounds))]
            bounds = np.array((data[:,0], data[:,-1])).T
        else:
            # if bounds are not at the ends of the data,
            # consider only the inner parts of the data from here on
            data = data[:, (data[0] >= np.min(bounds[0])) &
                           (data[0] <= np.max(bounds[0]))]
            # make sure that bounds are actually part of the x range: 
            # keep their y values, put x on the closest existing point
            bounds[0, 0] = data[0, 0]
            bounds[0, 1] = data[0, -1]

    # ensure that the 'left' value of the data is higher than the 'right'
    # NOTE: This is insensitive to whether the energy axis is binding or
    # kinetic, but WILL give unphysical results where the background goes
    # 'up' without complaining if that's what's in the data!
    if data[1, 0] < data[1, -1]:
        is_reversed = True
        data = data[:,::-1]
    else:
        is_reversed = False
    # make the bounds follow the same order as the data;
    # i.e. if kinetic energy -> lower value first, otherwise higher first
    if (np.sign(bounds[0, 0] - bounds[0, -1])
            != np.sign(data[0, 0] - data[0, -1])):
        bounds = bounds[:, ::-1]

    # Initial value of the background shape B. The total background S = bounds[1,1] + B,
    # and B is initially zero
    B = data[1] * 0

    for it in range(maxit):
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        # background-subtracted y so far, and cumulative integral:
        y_sub = data[1] - B - bounds[1, 1]
        y_int = cumulative_trapezoid(y_sub[::-1], data[0, ::-1], initial=0)[::-1]
        # Calculate new k = (yl - yr) / (integral of y over the whole range)
        k = (bounds[1, 0] - bounds[1, 1]) / y_int[0]
        # new B is simply the cumulative integral normalized by the new k
        B_new = k*y_int
        # If B_new is close to B, exit.
        if np.sum((B - B_new)**2) / len(B) < tol:
            B = np.copy(B_new)
            break
        else:
            B = np.copy(B_new)
    else:
        print("Max iterations exceeded before convergence.")
    B += bounds[1, 1]
    if is_reversed:
        B = B[::-1]
        data = data[:,::-1]

    # check the original data range, fill up the missing parts
    npx = np.array(x)
    index_exists = np.where((npx >= np.min(data[0])) & 
                            (npx <= np.max(data[0])))[0]
    B_whole_range = np.concatenate((
        np.full(index_exists[0], B[0]),
        B,
        np.full(len(npx) - index_exists[-1] - 1, B[-1])
        ))
    return B_whole_range


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
