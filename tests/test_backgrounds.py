import matplotlib.pyplot as plt
import numpy as np
import lmfit
from ..lmfitxps.backgrounds import tougaard
dat = np.loadtxt('../examples/NIST_Gauss3.dat')
x = dat[:, 1]
y = dat[:, 0]
plt.plot(x,y)
plt.show()
