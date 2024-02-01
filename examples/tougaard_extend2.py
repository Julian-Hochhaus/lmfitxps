import numpy as np
import matplotlib.pyplot as plt
import lmfit
import os
from examples.context import models
from examples.context import backgrounds
exec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = np.genfromtxt(exec_dir + '/examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
output_dir = os.path.join(exec_dir, 'examples', 'plots')
os.makedirs(output_dir, exist_ok=True)
calc_tougaard=backgrounds.tougaard
delta_x=(x[-1]-x[0])/len(x)
len_padded = int(33 / delta_x)
padded_x = np.concatenate((np.linspace( x[0] - delta_x * len_padded,x[0]-delta_x, len_padded),x))
padded_y = np.concatenate((np.mean(y[:10]) * np.ones(len_padded), y))
tg=calc_tougaard(padded_x,padded_y, 219,144.506, 0.281, 268.598)
fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
ax1.plot(padded_x, tg, label='Best Fit')
ax1.plot(x, y, 'x', markersize=4, label='Data Points')
ax1.legend()
ax1.set_xlabel('energy in eV')
ax1.set_ylabel('intensity in arb. units')
ax1.tick_params(axis='x', which='both',top=True, direction='in')
ax1.tick_params(axis='y', which='both', right=True,direction='in')
ax2.tick_params(axis='x', which='both',top=True, direction='in')
ax2.tick_params(axis='y', which='both', right=True, direction='in')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
plt.subplots_adjust(hspace=0)
ax2.set_xlabel('x')
ax2.set_ylabel('Residual')

plot_filename = os.path.join(output_dir, f'plot_extend.png')
plt.savefig(plot_filename, dpi=300)
plt.close(fig)