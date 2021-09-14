import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma


def make_gammas(ks, thetas, x):
    res = {}
    ordered_keys = []
    for theta in sorted(thetas):
        for k in sorted(ks):
            name = '$k$=' + str(k) + '; $\\theta$=' + str(theta)
            res[name] = gamma.pdf(x, k, scale=theta)
            ordered_keys.append(name)

    return pd.DataFrame(res)[ordered_keys]

gamma_df_k = make_gammas([1,2,3,5,7,9], [2], np.linspace(0,25, num=100))
gamma_df_theta = make_gammas([2],[1,2,5,7,11,13], np.linspace(0,25, num=100))

fig, axarr = plt.subplots(1,2,figsize=(15,5), sharey=True)
gamma_df_k.plot(ax=axarr[0], fontsize=16)
axarr[0].legend(fontsize=14)
gamma_df_theta.plot(ax=axarr[1], fontsize=16)
axarr[1].legend(fontsize=14)
plt.suptitle('Gamma distributions for various $k$ and $\\theta$ values', fontsize=18)
plt.show()

import numpy as np
def gamma_mom(x):
    avg = np.mean(x)
    var = np.var(x)
    k = avg ** 2 / var
    theta = var / avg

    return k, theta

def gamma_bootstrap_estimate(true_k, true_theta, sample_size=[50, 100, 1000], draws=100, method='all'):
    true_mean = true_k * true_theta
    true_var = true_k * true_theta ** 2

    result = []
    for this_many in sample_size:

        # Generate this_many samples from the true Gamma
        rvs = [gamma.rvs(true_k, scale=true_theta, size=this_many)
               for n in range(draws)]

        if method == 'all' or method == 'scipy':
            estimates_scipy = (gamma.fit(x, floc=0) for x in rvs)
            (k_scipy, loc_scipy, theta_scipy) = zip(*estimates_scipy)
            result.append({'sample_size': this_many, 'k_estimate': k_scipy, 'theta_estimate': theta_scipy,
                           'true_theta': true_theta, 'true_k': true_k, 'method': 'scipy'})

        if method == 'all' or method == 'mom':
            estimates_mom = (gamma_mom(x) for x in rvs)
            (k_mom, theta_mom) = zip(*estimates_mom)
            result.append({'sample_size': this_many, 'k_estimate': k_mom, 'theta_estimate': theta_mom,
                           'true_theta': true_theta, 'true_k': true_k, 'method': 'mom'})

    return pd.concat([pd.DataFrame(r) for r in result])

true_k = 2
true_theta = 2
num_samples = [10, 25, 50, 100, 500, 1000]
num_draws = 1000
estimates_for_one_k = gamma_bootstrap_estimate(true_k, true_theta, num_samples, draws=num_draws)
estimates_for_one_k.head()

import seaborn as sns
sns.set(style="ticks")
sns.set_context("poster") # this helps when converting to static html for blog

plt.figure(figsize=(15,7))
ax = sns.violinplot(x='sample_size', y='k_estimate', data=estimates_for_one_k, hue='method', palette='muted',
               inner='quartile', split=True, hue_order=['mom', 'scipy'], linewidth=1)
sns.despine(offset=10, trim=True)
title_str = 'Estimates of k from ' + str(num_draws) + \
    ' bootstrap draws; true k=' + str(true_k) + \
    ', true $\\theta=$' + str(true_theta)
plt.title(title_str)
plt.show()

df_list = []
theta_val = 2
for k in [1,2,3,5,7,9]:
    tmp = gamma_bootstrap_estimate(k,theta_val, sample_size=num_samples, draws=num_draws)
    df_list.append(tmp)

big_df = pd.concat(df_list)
big_df['fractional_error'] = (big_df['k_estimate'] - big_df['true_k'] ) / big_df['true_k']
big_df.head()

import matplotlib.pyplot as plt
from matplotlib import gridspec

true_k = big_df['true_k'].unique()
num_k = len(true_k)
ncol=3
nrow= int(num_k / ncol)

sns.set(style="ticks")
sns.set_context("poster")
f, axarr = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(15,14))
for row in range(nrow):
    for col in range(ncol):
        idx = row * ncol + col
        this_ax = axarr[row,col]
        sns.boxplot(ax=this_ax, x="sample_size", y="fractional_error", hue='method', hue_order=['mom', 'scipy'],
                    data=big_df[big_df['true_k'] == true_k[idx] ], palette="muted",
                    showfliers=False,  linewidth=1, showmeans=True, meanline=True)
        this_ax.set_title('k='+str(true_k[idx]))
        if row == 0:
            this_ax.set_xlabel('')
        if col > 0:
            this_ax.set_ylabel('')

sns.despine(offset=10, trim=True)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.suptitle('Fractional estimation error across k, for $\\theta$='+str(theta_val))
plt.show()

q = [0.05,0.1,0.2,0.3, 0.4,0.5,0.6,0.7,0.8,0.9, 0.95, 0.975]
big_df['abs_fractional_error'] = big_df['fractional_error'].abs()
grouped = big_df.groupby(['method', 'sample_size', 'true_k'])
grouped_quantiles = grouped['abs_fractional_error'].quantile(q)
grouped_quantiles.name='fractional_error_quantile'

grouped_quantiles.index.names = map(lambda n: n if n is not None else 'confidence_level', grouped_quantiles.index.names)

quantiles_df = pd.DataFrame( pd.DataFrame(grouped_quantiles).to_records() )
quantiles_df.head()

sns.set(style="darkgrid")
sns.set_context("poster")
f = plt.figure(figsize=(17, 15))
g = sns.FacetGrid(quantiles_df, despine=True, sharey=False, col_wrap=3, col='sample_size', height=4, legend_out=True,
                  hue='method', hue_order=['mom', 'scipy'], margin_titles=True, palette='muted', xlim=[-0.1, 1.1])
g.map(plt.scatter, "confidence_level", "fractional_error_quantile")

conf_level = 0.95
for ax in g.axes.flat:
    ylims = ax.get_ylim()
    new_ylims = [-0.01, ylims[1]]
    ax.set_ylim(new_ylims)
    ax.plot([conf_level, conf_level], new_ylims, 'r--', alpha=0.5, linewidth=1)

sns.despine(offset=10, trim=True)
g.add_legend()
g.fig.subplots_adjust(wspace=0.4, hspace=0.5)
plt.show()


