import numpy as np

import numpy as np

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
    # Make sure we've been passed arrays and not lists.
    x = np.array(x)
    y = np.array(y)

    # Sanity check: Do we actually have data to process here?
    if not (np.any(x) and np.any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return np.asarray(x * 0)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = np.abs(y - y.max()).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print("Boundaries too high for algorithm: returning a zero background.")
        return np.asarray(x * 0)

    # Locate the minima either side of maxidx.
    lmidx = np.abs(y[0:maxidx] - y[0:maxidx].min()).argmin()
    rmidx = np.abs(y[maxidx:] - y[maxidx:].min()).argmin() + maxidx

    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = y * 0
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    it = 0
    while it < maxit:
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1] - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum

        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] + y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum

        # If Bnew is close to B, exit.
        B = Bnew - B
        if (B ** 2).sum() < tol ** 2:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("Max iterations exceeded before convergence.")

    if is_reversed:
        return np.asarray((yr + B)[::-1])
    else:
        return np.asarray(yr + B)



def tougaard_calculate(x, y, tb=2866, tc=1643, tcd=1, td=1, maxit=100):
    """
    Calculate the Tougaard background for a given set of x and y data. Inspired from https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/surface/people/james_mudd/igor/

    Args:
        x (array-like): The x-values of the data.
        y (array-like): The y-values of the data.
        tb (float, optional): Initial background value. Defaults to 2866.
        tc (float, optional): C parameter. Defaults to 1643.
        tcd (float, optional): C' parameter. Defaults to 1.
        td (float, optional): D parameter. Defaults to 1.
        maxit (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        list: The Tougaard background calculated from the input data along with the final tb value.
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

    return [np.asarray(y[len(y) - 1] + Btou), tb]
