import numpy as np
import matplotlib.pyplot as plt
import lmfit
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lmfitxps import models
import matplotlib as mpl

exec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fermi = models.FermiEdgeModel(prefix='fermi_', independent_vars=["x"])
const = lmfit.models.ConstantModel(prefix='const_')
fit_model=fermi+const
data = np.genfromtxt(exec_dir + '/examples/FermiEdge.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
output_dir = os.path.join(exec_dir, 'examples/', 'plots')
os.makedirs(output_dir, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
fig.patch.set_facecolor('#FCFCFC')
params = lmfit.Parameters()
params.add('fermi_amplitude', value=232)
params.add('fermi_sigma', value=0.2126)
params.add('fermi_kt', value=0.0026, vary=False)
params.add('fermi_center', value=236)
params.add('const_c', value=np.min(y))


result = fit_model.fit(y, params, y=y, x=x, weights=1 /(np.sqrt(y)))
comps = result.eval_components(x=x, y=y)
print(result.fit_report())

cmap = mpl.colormaps['tab20']
ax1.plot(x, result.best_fit, label='Best Fit', color=cmap(0))
ax1.plot(x, y, 'x', markersize=4, label='Data Points', color=cmap(2))

ax1.plot(x, comps['const_'], label='const. background', color='black')
ax1.plot(x, comps['fermi_'] + comps['const_'], color=cmap(4), label="fermi-edge")

ax1.fill_between(x, comps['fermi_'] + comps['const_'], comps['const_'], alpha=0.5,color=cmap(5))
ax1.legend()
ax1.set_xlabel('kin. energy in eV')
ax1.set_ylabel('intensity in arb. units')
# Set ticks only inside
ax1.tick_params(axis='x', which='both',top=True, direction='in')
ax1.tick_params(axis='y', which='both', right=True,direction='in')
ax2.tick_params(axis='x', which='both',top=True, direction='in')
ax2.tick_params(axis='y', which='both', right=True, direction='in')

ax1.set_yticklabels([])
ax1.set_title(f'FermiEdgeModel using kin. energy scale')
fig.subplots_adjust(hspace=0)
ax1.set_xlim(np.min(x), np.max(x))
# Residual plot
residual = result.residual
ax2.plot(x, residual)
ax2.set_xlabel('kin. energy in eV')
ax2.set_ylabel('Residual')

plot_filename = os.path.join(output_dir, f'plot_fermi_kin.png')
fig.savefig(plot_filename, dpi=300)
plt.close(fig)


fermi = models.FermiEdgeModel(prefix='fermi_', independent_vars=["x"])
const = lmfit.models.ConstantModel(prefix='const_')
fit_model=fermi+const
data = np.genfromtxt(exec_dir + '/examples/FermiEdge.csv', delimiter=',', skip_header=1)
x = 240-data[:, 0]
y = data[:, 1]
output_dir = os.path.join(exec_dir, 'examples/', 'plots')
os.makedirs(output_dir, exist_ok=True)

fig2, (ax21, ax22) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
fig2.patch.set_facecolor('#FCFCFC')
params = lmfit.Parameters()
params.add('fermi_amplitude', value=232, min=220, max=240)
params.add('fermi_sigma', value=0.15, min=0.12,max=0.17)
params.add('fermi_kt', value=0.0026, vary=False)
params.add('fermi_center', value=3.5, min=3, max=4)
params.add('const_c', value=np.min(y), min=np.min(y), max=np.mean(y))


result = fit_model.fit(y, params, y=y, x=x, weights=1 /(np.sqrt(y)))
comps = result.eval_components(x=x, y=y)
print(result.fit_report())

cmap = mpl.colormaps['tab20']
ax21.plot(x, result.best_fit, label='Best Fit', color=cmap(0))
ax21.plot(x, y, 'x', markersize=4, label='Data Points', color=cmap(2))

ax21.plot(x, comps['const_'], label='const. background', color='black')
ax21.plot(x, comps['fermi_'] + comps['const_'], color=cmap(4), label="fermi-edge")

ax21.fill_between(x, comps['fermi_'] + comps['const_'], comps['const_'], alpha=0.5,color=cmap(5))
ax21.legend()
ax21.set_xlabel('bin. energy in eV')
ax21.set_ylabel('intensity in arb. units')
# Set ticks only inside
ax21.tick_params(axis='x', which='both',top=True, direction='in')
ax21.tick_params(axis='y', which='both', right=True,direction='in')
ax22.tick_params(axis='x', which='both',top=True, direction='in')
ax22.tick_params(axis='y', which='both', right=True, direction='in')

ax21.set_yticklabels([])
ax21.set_title(f'FermiEdgeModel using bin. energy scale')
fig2.subplots_adjust(hspace=0)
ax21.set_xlim(np.min(x), np.max(x))
# Residual plot
residual = result.residual
ax22.plot(x, residual)
ax22.set_xlabel('bin. energy in eV')
ax22.set_ylabel('residual')
ax22.set_xlim(np.max(x), np.min(x))

plot_filename = os.path.join(output_dir, f'plot_fermi_bin.png')
fig2.savefig(plot_filename, dpi=300)
plt.close(fig2)