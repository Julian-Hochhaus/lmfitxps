import numpy as np
import matplotlib.pyplot as plt
import lmfit
import os
from lmfitxps.models import ShirleyBG
from lmfitxps.models import ConvGaussianDoniachSinglett
import matplotlib as mpl
exec_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
shirley_model=ShirleyBG(prefix='shirley_', independent_vars=['y'])
s1=ConvGaussianDoniachSinglett(prefix='s1_')
s2=ConvGaussianDoniachSinglett(prefix='s2_')

data = np.genfromtxt(exec_dir+'/examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
x = data[150:-35, 0]
y = data[150:-35, 1]
params = lmfit.Parameters()

params.add('shirley_k', value=0.002)
params.add('shirley_const', value=y[-1])
params.add('s1_amplitude', value=56000, min=0)
params.add('s1_sigma', value=0.2, min=0)
params.add('s1_gamma', value=0.0, vary=False)
params.add('s1_gaussian_sigma', value=0.14, min=0)
params.add('s1_center', value=92.4)
params.add('s2_amplitude', value=56000, min=0)
params.add('s2_sigma', value=0.14, min=0, expr='s1_sigma')
params.add('s2_gamma', value=0.02, expr='s1_gamma')
params.add('s2_gaussian_sigma', value=0.2, min=0, expr='s1_gaussian_sigma')
params.add('s2_center', value=92.2)

fit_model=shirley_model+s1+s2
result = fit_model.fit(y, params, y=y, x=x)
comps = result.eval_components(x=x, y=y)

cmap = mpl.colormaps['Blues']
fig, ax = plt.subplots(figsize=(2, 2))
fig.patch.set_facecolor('#2980B9')
ax.plot(x, result.best_fit, color=cmap(0))
ax.plot(x[::4], y[::4], 'x', markersize=4, color=cmap(0.86), markeredgewidth=2.5)
ax.plot(x, comps['s1_'] + comps['shirley_'], color=cmap(0.1))
ax.plot(x, comps['s2_'] + comps['shirley_'], color=cmap(0.3))
ax.plot(x, comps['shirley_'], color=cmap(0.5), label='shirley')
ax.fill_between(x, comps['s1_'] + comps['shirley_'], comps['shirley_'], alpha=0.3, color=cmap(0.1))
ax.fill_between(x, comps['s2_'] + comps['shirley_'], comps['shirley_'], alpha=0.3, color=cmap(0.3))
ax.axis('off')
ax.set_xlim(91,94)
plt.savefig('docs/src/logo_spectrum.svg', format='svg', bbox_inches='tight')
plt.show()