import numpy as np
import matplotlib.pyplot as plt
import lmfit
import os
from examples.context import models
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
output_dir = os.path.join(exec_dir, 'examples', 'plots')
os.makedirs(output_dir, exist_ok=True)

# Create a combined plot
combined_fig, combined_ax = plt.subplots(figsize=(6, 4))

residual_fig, residual_ax = plt.subplots(figsize=(6, 4))

for j in [0]+[i for i in range(27,35,2)]:
    print(j)
    params = lmfit.Parameters()
    params.add('tougaard_B', value=148.969)
    params.add('tougaard_C', value=144.506, vary=False)
    params.add('tougaard_C_d', value=0.281, vary=False)
    params.add('tougaard_D', value=268.598, vary=False)
    params.add('tougaard_extend', value=j)
    params.add('d1_amplitude', value=71980, vary=False)
    params.add('d1_sigma', value=0.2126, vary=False)
    params.add('d1_gamma', value=0.01, vary=False)
    params.add('d1_gaussian_sigma', value=0.0892)
    params.add('d1_center', value=92.2273)
    params.add('d1_soc', value=3.67127)
    params.add('d1_height_ratio', value=0.7)
    params.add('d1_fct_coster_kronig', value=1.04, vary=False)
    params.add('d2_amplitude', value=56000, min=0)
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

    result = fit_model.fit(y, params, y=y, x=x)
    comps = result.eval_components(x=x, y=y)
    print(result.fit_report())
    fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    # Data plot
    ax1.plot(x, result.best_fit, label='Best Fit')
    ax1.plot(x, y, 'x', markersize=4, label='Data Points')
    if j == 0:
        ax1.plot(x, comps['const_'] + comps['tougaard_'], label='Tougaard + Const')
        combined_ax.plot(x, comps['const_']+comps['tougaard_'], label=f'j={j}')
    else:
        combined_ax.plot(x, comps['tougaard_'], label=f'j={j}')
        ax1.plot(x, comps['tougaard_'], label='Tougaard')
    ax1.legend()
    ax1.set_xlabel('energy in eV')
    ax1.set_ylabel('intensity in arb. units')
    # Set ticks only inside
    ax1.tick_params(axis='x', which='both',top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True,direction='in')
    ax2.tick_params(axis='x', which='both',top=True, direction='in')
    ax2.tick_params(axis='y', which='both', right=True, direction='in')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(f'Value of extend Parameter: {j}')
    plt.subplots_adjust(hspace=0)
    ax1.set_xlim(np.min(x), np.max(x))
    # Residual plot
    residual = result.residual
    ax2.plot(x, residual/np.sqrt(y), label='Residual')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('Residual')

    # Save individual plots
    plot_filename = os.path.join(output_dir, f'plot_{j}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


    # Add residual to the combined residual plot
    residual_ax.plot(x, residual/np.sqrt(y), label=f'j={j}')

# Save and close the combined plots
combined_ax.legend()
combined_ax.set_xlabel('x')
combined_ax.set_ylabel('y')
combined_plot_filename = os.path.join(output_dir, 'combined_plot.png')
combined_fig.savefig(combined_plot_filename, dpi=300)
plt.close(combined_fig)

residual_ax.legend()
residual_ax.set_xlabel('x')
residual_ax.set_ylabel('Residual')
residual_plot_filename = os.path.join(output_dir, 'combined_residual_plot.png')
residual_fig.savefig(residual_plot_filename, dpi=300)
plt.close(residual_fig)
