import numpy as np
import matplotlib.pyplot as plt
import lmfit
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lmfitxps import models
import matplotlib as mpl
def guess_extend(x,y, B, C, C_d, D):
    y_end=np.mean(y[-10:])
    delta_x=(x[-1]-x[0])/len(x)
    tougaard_sum=0
    n=0
    lower=0
    lower_reached=False
    while B*tougaard_sum<y_end*0.9:
        n+=1
        tougaard_sum=0
        for j in range(n):
             tougaard_sum+= (j*delta_x) / ((C + C_d * (j*delta_x) ** 2) ** 2
                                              + D * (j*delta_x) ** 2) * y_end * delta_x
        if B*tougaard_sum>=y_end*0.5 and not lower_reached:
            lower=n
            lower_reached=True
    print('result, n={}, extend={}, lower={}'.format(n, n*delta_x, lower*delta_x))

exec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
tougaard_model = models.TougaardBG(prefix='tougaard_', independent_vars=['y', 'x'])
d1 = models.ConvGaussianDoniachDublett(prefix='d1_')
d2 = models.ConvGaussianDoniachDublett(prefix='d2_')
const = lmfit.models.ConstantModel(prefix='const_')

data = np.genfromtxt(exec_dir + '/examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
guess_extend(x,y,156.969,144.506,0.281,268.598)
output_dir = os.path.join(exec_dir, 'docs/src/', 'plots')
os.makedirs(output_dir, exist_ok=True)

combined_fig, (combined_ax1, combined_ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
combined_fig.patch.set_facecolor('#FCFCFC')
residual_fig, residual_ax = plt.subplots(figsize=(6, 4))
residual_fig.patch.set_facecolor('#FCFCFC')
combined2_fig, combined2_ax = plt.subplots(figsize=(6, 4))
combined2_fig.patch.set_facecolor('#FCFCFC')
tg_bgs=[]
tg_res=[]
for j in [0]+[i for i in range(27,35,1)]:
    params = lmfit.Parameters()
    params.add('tougaard_B', value=148.969)
    params.add('tougaard_C', value=144.506, vary=False)
    params.add('tougaard_C_d', value=0.281, vary=False)
    params.add('tougaard_D', value=268.598, vary=False)
    params.add('tougaard_extend', value=j)
    params.add('d1_amplitude', value=71980, vary=False)
    params.add('d1_sigma', value=0.2126, vary=False)
    params.add('d1_gamma', value=0.01, vary=False)
    params.add('d1_gaussian_sigma', value=0.0892, vary=False)
    params.add('d1_center', value=92.2273, vary=False)
    params.add('d1_soc', value=3.67127, vary=False)
    params.add('d1_height_ratio', value=0.7, vary=False)
    params.add('d1_fct_coster_kronig', value=1.04, vary=False)
    params.add('d2_amplitude', value=43966, vary=False)
    params.add('d2_sigma', value=0.2, expr='d1_sigma')
    params.add('d2_gamma', value=0.0, expr='d1_gamma')
    params.add('d2_gaussian_sigma', value=0.14, expr='d1_gaussian_sigma')
    params.add('diff', value=0.31)
    params.add('d2_center', value=92.4, expr='d1_center+diff')
    params.add('d2_soc', value=3.67, expr="d1_soc")
    params.add('d2_height_ratio', value=0.7, expr='d1_height_ratio')
    params.add('d2_fct_coster_kronig', value=1, expr='d1_fct_coster_kronig')

    if j == 0:
        params.add('const_c', value=y[-1])
        fit_model = tougaard_model + d1+d2  + const
    else:
        fit_model = tougaard_model + d1+d2

    result = fit_model.fit(y, params, y=y, x=x, weights=1 /(np.sqrt(y)))
    comps = result.eval_components(x=x, y=y)
    fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    fig.patch.set_facecolor('#FCFCFC')
    # Data plot
    cmap = mpl.colormaps['tab20']
    ax1.plot(x, result.best_fit, label='Best Fit', color=cmap(0))
    ax1.plot(x, y, 'x', markersize=4, label='Data Points', color=cmap(2))
    if j == 0:
        ax1.plot(x, comps['const_'] + comps['tougaard_'], label='Tougaard + Const', color='black')
        ax1.plot(x, comps['d1_'] + comps['const_'] + comps['tougaard_'], color=cmap(4), label="bulk")
        ax1.plot(x, comps['d2_'] + comps['const_'] + comps['tougaard_'], color=cmap(6), label="surface")
        ax1.fill_between(x, comps['d1_'] + comps['const_'] + comps['tougaard_'], comps['const_'] +comps['tougaard_'], alpha=0.5,color=cmap(5))
        ax1.fill_between(x, comps['d2_'] + comps['const_'] + comps['tougaard_'], comps['const_'] +comps['tougaard_'], alpha=0.5,color=cmap(7))
        tg_bgs.append([comps['const_']+comps['tougaard_'], 0])
        tg_res.append([result.residual,0])
    else:
        if j>=28 and j<=30:
            tg_bgs.append([comps['tougaard_'], j])
            tg_res.append([result.residual, j])
        ax1.plot(x, comps['d1_'] + comps['tougaard_'], color=cmap(4), label="bulk")
        ax1.plot(x, comps['d2_'] + comps['tougaard_'], color=cmap(6), label="surface")
        ax1.fill_between(x, comps['d1_'] + comps['tougaard_'], comps['tougaard_'], alpha=0.5, color=cmap(5))
        ax1.fill_between(x, comps['d2_'] + comps['tougaard_'], comps['tougaard_'], alpha=0.5, color=cmap(7))
        ax1.plot(x, comps['tougaard_'], label='Tougaard')
    ax1.legend()
    ax1.set_xlabel('energy in eV')
    ax1.set_ylabel('intensity in arb. units')
    # Set ticks only inside
    ax1.tick_params(axis='x', which='both',top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True,direction='in')
    ax2.tick_params(axis='x', which='both',top=True, direction='in')
    ax2.tick_params(axis='y', which='both', right=True, direction='in')
    #ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(f'extend={j}; red. chi-squared='+"{:.2f}".format(result.redchi))
    plt.subplots_adjust(hspace=0)
    ax1.set_xlim(np.min(x), np.max(x))
    # Residual plot
    residual = result.residual
    ax2.plot(x, residual, label='Residual')
    ax2.legend()
    ax2.set_xlabel('energy in eV')
    ax2.set_ylabel('Residual')

    # Save individual plots
    plot_filename = os.path.join(output_dir, f'plot_{j}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


    # Add residual to the combined residual plot
    residual_ax.plot(x, residual/np.sqrt(y), label=f'j={j}')

for i, item in enumerate(tg_bgs[1:], start=1):
    print(i, item[1])
    combined_ax1.plot(x, item[0] - tg_bgs[0][0], label='extend={}'.format(item[1]), color=cmap(i))

for i, item in enumerate(tg_res[1:], start=1):
    combined_ax2.plot(x, (item[0] - tg_res[0][0]), label='extend={}'.format(item[1]), color=cmap(i))
combined_ax1.legend()
combined_ax2.set_xlabel('energy in eV')
combined_ax1.set_ylabel('intensity in arb. units')
combined_ax1.set_title(r'$B_T(extend)-(B_T(extend=0)+B_C)$')
combined_ax2.set_title(r'$Res(extend)$/$Res(extend=0)$')
combined_plot_filename = os.path.join(output_dir, 'combined_plot.png')
combined_fig.savefig(combined_plot_filename, dpi=300)
plt.close(combined_fig)


for item in tg_bgs:
    combined2_ax.plot(x, item[0], label='extend={}'.format(item[1]))
combined2_ax.legend()
combined2_ax.set_ylim(2600,2800)
combined2_ax.set_xlim(94.5,np.max(x))
combined2_ax.set_xlabel('energy in eV')
combined2_ax.set_ylabel('intensity in arb. units')
combined2_ax.set_title(r'$B_T(extend)-(B_T(extend=0)+B_C)$')
combined2_plot_filename = os.path.join(output_dir, 'combined2_plot.png')
combined2_fig.savefig(combined2_plot_filename, dpi=300)
plt.close(combined2_fig)

residual_ax.legend()
residual_ax.set_xlabel('x')
residual_ax.set_ylabel('Residual')
residual_plot_filename = os.path.join(output_dir, 'combined_residual_plot.png')
residual_fig.savefig(residual_plot_filename, dpi=300)
plt.close(residual_fig)

# with binding energy
data = np.genfromtxt(exec_dir + '/examples/clean_Au_4f.csv', delimiter=',', skip_header=1)

x = 180-data[:, 0]
y = data[:, 1]
output_dir = os.path.join(exec_dir, 'docs/src/', 'plots')
os.makedirs(output_dir, exist_ok=True)




params = lmfit.Parameters()
params.add('tougaard_B', value= 197.643926)
params.add('tougaard_C', value=144.506, vary=False)
params.add('tougaard_C_d', value=0.281, vary=False)
params.add('tougaard_D', value=268.598, vary=False)
params.add('tougaard_extend', value=0)
params.add('d1_amplitude', value=71980)
params.add('d1_sigma', value=0.21)
params.add('d1_gamma', value=0.01)
params.add('d1_gaussian_sigma', value=0.0892)
params.add('d1_center', value=180-92.2273)
params.add('d1_soc', value=3.67127)
params.add('d1_height_ratio', value=0.7)
params.add('d1_fct_coster_kronig', value=1.04, vary=False)
params.add('d2_amplitude', value=43966)
params.add('d2_sigma', value=0.2, expr='d1_sigma')
params.add('d2_gamma', value=0.0, expr='d1_gamma')
params.add('d2_gaussian_sigma', value=0.14, expr='d1_gaussian_sigma')
params.add('diff', value=-0.31165)
params.add('d2_center', value=180-92.4, expr='d1_center+diff')
params.add('d2_soc', value=-3.67, expr="d1_soc")
params.add('d2_height_ratio', value=0.7, expr='d1_height_ratio')
params.add('d2_fct_coster_kronig', value=1, expr='d1_fct_coster_kronig')
params.add('const_c', value=2677.97771)
fit_model = tougaard_model + d1+d2  + const

result = fit_model.fit(y, params, y=y, x=x, weights=1 /(np.sqrt(y)))
comps = result.eval_components(x=x, y=y)
fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
fig.patch.set_facecolor('#FCFCFC')
# Data plot
cmap = mpl.colormaps['tab20']
ax1.plot(x, result.init_fit, label='Init Fit', color=cmap(3))
ax1.plot(x, result.best_fit, label='Best Fit', color=cmap(0))
ax1.plot(x, y, 'x', markersize=4, label='Data Points', color=cmap(2))
ax1.plot(x, comps['const_'] + comps['tougaard_'], label='Tougaard + Const', color='black')
ax1.plot(x, comps['d1_'] + comps['const_'] + comps['tougaard_'], color=cmap(4), label="bulk")
ax1.plot(x, comps['d2_'] + comps['const_'] + comps['tougaard_'], color=cmap(6), label="surface")
ax1.fill_between(x, comps['d1_'] + comps['const_'] + comps['tougaard_'], comps['const_'] +comps['tougaard_'], alpha=0.5,color=cmap(5))
ax1.fill_between(x, comps['d2_'] + comps['const_'] + comps['tougaard_'], comps['const_'] +comps['tougaard_'], alpha=0.5,color=cmap(7))

ax1.legend()
ax1.set_xlabel('energy in eV')
ax1.set_ylabel('intensity in arb. units')
# Set ticks only inside
ax1.tick_params(axis='x', which='both',top=True, direction='in')
ax1.tick_params(axis='y', which='both', right=True,direction='in')
ax2.tick_params(axis='x', which='both',top=True, direction='in')
ax2.tick_params(axis='y', which='both', right=True, direction='in')
#ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title(f'binding energy; red. chi-squared='+"{:.2f}".format(result.redchi))
plt.subplots_adjust(hspace=0)
ax1.set_xlim(np.max(x), np.min(x))
# Residual plot
residual = result.residual
ax2.plot(x, residual, label='Residual')
ax2.legend()
ax2.set_xlabel('binding energy in eV')
ax2.set_ylabel('Residual')

# Save individual plots
plot_filename = os.path.join(output_dir, f'plot_binding_energy.png')
plt.savefig(plot_filename, dpi=300)
#plt.show()
plt.close(fig)



residual_ax.legend()
residual_ax.set_xlabel('x')
residual_ax.set_ylabel('Residual')
residual_plot_filename = os.path.join(output_dir, 'combined_residual_plot.png')
residual_fig.savefig(residual_plot_filename, dpi=300)
plt.close(residual_fig)