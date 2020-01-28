import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc


def plot_period(obs_nu, obs_power, allpds, percentile):
    rc('text', usetex=True)
    rc('font', size=18)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(1. / obs_nu, obs_power, label=r'obs')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 99, axis=0), label=r'99\%-sim')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 95, axis=0), label=r'95\%-sim')
    # ax1.plot(1. / obs_nu, np.max(allpds, axis=0), label='max-sim')
    print(1. / obs_nu[np.where(obs_power > np.percentile(allpds, percentile, axis=0))])
    plt.xlabel(r'\textbf{periods} (d)')
    plt.ylabel(r'$ \mathcal{{LSP}} $ \textbf{power}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fig