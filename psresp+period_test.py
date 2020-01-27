import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from astropy.stats import LombScargle


def _spectrum(x, slope):
    y = x ** slope

    return y


def _timmerlc(slope, nt='None', dt='None', mean='None', sigma='None', seed='None'):
    if dt == 'None':
        dt = 1
    if nt == 'None':
        nt = 65536
    if mean == 'None':
        mean = 0
    if sigma == 'None':
        sigma = 1
    if seed == 'None':
        seed = 42

    simfreq = np.linspace(1, nt / 2 - 1, num=nt / 2, dtype='float64') / (dt * nt)
    simpsd = _spectrum(simfreq, slope)
    fac = np.sqrt(simpsd)

    pos_real = np.random.RandomState(seed).normal(size=int(nt / 2)) * fac
    pos_imag = np.random.RandomState(seed).normal(size=int(nt / 2)) * fac

    pos_imag[int(nt / 2) - 1] = 0

    if float(nt / 2.) > int(nt / 2):
        neg_real = pos_real[0:int(nt / 2)][::-1]
        neg_imag = -pos_real[0:int(nt / 2)][::-1]
    else:
        neg_real = pos_real[0:int(nt / 2) - 1][::-1]
        neg_imag = -pos_real[0:int(nt / 2) - 1][::-1]

    real = np.hstack((0., pos_real, neg_real))
    imag = np.hstack((0., pos_imag, neg_imag))

    arg = real + 1j * imag
    rate = np.fft.ifft(arg).real
    time = dt * np.linspace(0, nt - 1, nt, dtype='float')

    avg = np.mean(rate)
    std = np.sqrt(np.var(rate))

    rate = (rate - avg) * sigma / std + mean

    return time, rate


def _psresp_period(t, y, dy, slopes, binning, number_simulations, oversampling):
    '''
    This is an adaptation of the PSRESP method optimised for a test example of period detection with the LombScargle.
    The noise power is fixed, the SUF is not calculated.
    :param t: array of time
    :param y: array of magnitude
    :param dy: array of errors
    :param slopes: array of slopes for power noise to check
    :param binning: array of time series binning for PSRESP
    :param number_simulations: float for Monte Carlo simulation
    :param oversampling: float for Monte Carlo simulation
    :return: periodogram
    '''
    from astropy.stats import LombScargle

    bin = binning
    date, rat, raterr = t, y, dy
    date = date - date[0]
    duration = np.max(date) - np.min(date)
    npoints = int(duration / bin) * number_simulations * oversampling
    params = -slopes
    lc_variance = np.var(rat) - np.var(raterr)

    # observed PDS calculation
    # obs_nu, obs_pds = _do_periodogram(rat)
    # obs_nu = np.log10(obs_nu)
    obs_nu = 1. / np.arange(1, 100 + 1, 1)
    obs_pds = LombScargle(date, rat, raterr).power(obs_nu)

    # normalisation
    # obs_pds = (2. * duration) / (np.mean(rat) * np.mean(rat) * len(rat) * len(rat)) * obs_pds

    # rebin
    # obs_freqs, obs_power = _rebinlc(obs_nu, obs_pds, dt=df)
    # obs_power = np.log10(obs_power)
    obs_freqs, obs_power = obs_nu, obs_pds

    # create fake light curve
    faketime, fakerate = _timmerlc(params, nt=npoints, dt=bin / oversampling)

    # calculate level of Poisson noise
    factor = ((len(raterr) / (2. * duration)) - (1. / duration))
    p_noise = np.sum(raterr ** 2.) / (len(raterr) * factor)

    # calculate high frequency aliased power
    uplim = 1. / (2. * bin)
    lowlim = 1. / (2. * (bin / 10))
    intfreq = np.empty([int((lowlim - uplim) / uplim) + 2])
    for i in range(len(intfreq)):
        intfreq[i] = uplim * (i + 1)
    intpds = _spectrum(intfreq, params)
    integral = np.trapz(intpds, x=intfreq)
    p_alias = integral / factor

    # long light curve is divided and resultant PDS are calculated
    allpds = np.empty([number_simulations, len(obs_freqs)])
    for j in range(number_simulations):
        print('computing PDS ' + str(j + 1))

        # indices for each segment
        lobin = int(duration * j / (bin / oversampling))
        hibin = int(duration * j / (bin / oversampling)) + int(duration / (bin / oversampling))

        # taken from appropriate section of light curve
        temptime = faketime[lobin:hibin]
        temprate = fakerate[lobin:hibin]

        # shift start time to zero
        temptime = temptime - temptime[0]

        # set bintime equal to original light curve time
        bintime = date
        binrate = np.interp(date, temptime, temprate)

        # rescale simulated LC to the mean and variance of the original
        tempvar = np.sqrt(np.var(binrate))
        binrate = (binrate - np.mean(binrate)) * ((np.sqrt(lc_variance)) / tempvar) + np.mean(rat)
        # print(bintime)

        # calculate PDS of simulated light curve
        # sim_nu, sim_pds = _do_periodogram(binrate)
        # sim_nu = np.log10(sim_nu)
        # sim_pds = sim_pds + p_noise + p_alias
        sim_nu = obs_nu
        # print(1. / sim_nu)
        sim_pds = LombScargle(bintime, binrate, np.ones(len(bintime))).power(sim_nu)

        # normalisation
        # sim_pds = (2. * (np.max(bintime) - np.min(bintime))) / (np.mean(binrate)
        #                                                         * np.mean(binrate) * len(binrate) * len(
        #     binrate)) * sim_pds

        # rebin simulated PDS in same manner as observed
        # logfreqs, power = _rebinlc(sim_nu, sim_pds, dt=df)
        # logpower = np.log10(power)
        logfreqs, logpower = sim_nu, sim_pds

        # large array for later calculations of mean and rms error
        for k in range(len(logpower)):
            allpds[j, k] = logpower[k]

    avg_pds = np.empty([len(obs_freqs)])
    pds_err = np.empty([len(avg_pds)])
    for i in range(len(avg_pds)):
        avg_pds[i] = np.mean(allpds[:, i])
    for i in range(len(pds_err)):
        pds_err[i] = np.sqrt(np.var(allpds[:, i]))

    return faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds, sim_pds, obs_nu


def period(t, y, dy, slopes, binning, oversampling=10, number_simulations=100, percentile=99):
    """
    Detect periods exceeding the defined percentile of simulated periodograms with the same noise level as the data set.
    To detect the noise level, the PSRESP method is used.
    The artificial light curves are generated using the algorithm by Timmer and Koenig (1995).
    For an introduction to the PSRESP method, see Uttley (2002).
    The function returns a results dictionary with the following content:
    - ``periods`` (`~numpy.ndarray`) -- periods exceeding the percentile
    Parameters
    ----------
    t : `~numpy.ndarray`
        Time array of the light curve
    y : `~numpy.ndarray`
        Flux array of the light curve
    dy : `~numpy.ndarray`
        Flux error array of the light curve
    slopes : `~numpy.ndarray`
        slopes of the power law model
    binning : `~numpy.ndarray`
        bin length for the light curve in units of ``t``
    percentile : `float`
        percentile of the distribution of success fraction, `0 < significance < 1`
    oversampling : `int`
        oversampling factor of the simulated light curve, default is 10
    number_simulations : `int`
        number of simulations for each model parameter, default is 10
    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).
    References
    ----------
    .. [1] Timmer and Koenig (1995), "On generating power law noise",
       `Link <http://adsabs.harvard.edu/abs/1995A%26A...300..707T>`_
    .. [2] Uttley et al, "Measuring the broad-band power spectra of active galactic nuclei with RXTE",
       `Link <https://academic.oup.com/mnras/article/332/1/231/974626/Measuring-the-broad-band-power-spectra-of-active>`_
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=18)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    t = t - t[0]

    # psresp
    print('begin psresp')
    faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds, sim_pds, obs_nu = \
        _psresp_period(t, y, dy, slopes, binning, number_simulations, oversampling)

    # plot of observed and simulated periodograms respecting the percentile
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(1. / obs_nu, obs_power, label=r'obs')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 99, axis=0), label=r'99\%-sim')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 95, axis=0), label=r'95\%-sim')
    # ax1.plot(1. / obs_nu, np.max(allpds, axis=0), label='max-sim') # print max distribution
    print(1. / obs_nu[np.where(obs_power > np.percentile(allpds, percentile, axis=0))])
    plt.xlabel(r'\textbf{periods} (d)')
    plt.ylabel(r'$ \mathcal{{LSP}} $ \textbf{power}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return dict(periods=1. / obs_nu[np.where(obs_power > np.percentile(allpds, percentile, axis=0))])


print('go')

# set up test parameters
test_case = dict(period=16,  # location of periodicity
                 amplitude=.2,  # amplitude of periodicity
                 slope=1.6,  # power of noise
                 nt=1000,  # initial length of test data
                 oversampling=10,  # oversampling to reduce red noise in simulation
                 binning=10,  # binning of the simulation
                 subset=400,  # subset to reduce computation time, resulting in subset+1 data points
                 # the following is not used in this example because these parameters are fixed
                 # slopes=np.linspace(0.5, 2.5, 21),  # slopes to check
                 # dt=np.array([2, 3, 4, 5, 6]),  # binning of time series
                 # df=np.array([0.3, 0.5, 0.7, 0.9, 1.1]),  # binning of periodogram
                 n_simulations=100,  # number of simulated time series
                 percentile=99,  # percentile for significance
                 )

# create test data
# power law noise
t_ini, y_ini, = _timmerlc(-test_case['slope'], test_case['nt'], test_case['binning'] / test_case['oversampling'])
# periodicity
y_ini = y_ini + test_case['amplitude'] * np.sin(2 * np.pi * t_ini / test_case['period'])
# errors
dy = np.ones([test_case['nt']])
# subset
t, y, dy = t_ini[int((test_case['nt'] - test_case['subset']) / 2):int((test_case['nt'] + test_case['subset']) / 2) + 1], \
           y_ini[int((test_case['nt'] - test_case['subset']) / 2):int((test_case['nt'] + test_case['subset']) / 2) + 1], \
           dy[int((test_case['nt'] - test_case['subset']) / 2):int((test_case['nt'] + test_case['subset']) / 2) + 1]

print('test set ready')

# test on LombScargle, limited on  max period = 100 for visualisation
obs_nu = 1. / np.arange(1, 100 + 1, 1)
obs_pds = LombScargle(t, y, dy).power(obs_nu)

# plot LombScargle
rc('text', usetex=True)
rc('font', size=18)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(1. / obs_nu, obs_pds, label=r'obs')
plt.xlabel(r'\textbf{periods} (d)')
plt.ylabel(r'$ \mathcal{{LSP}} $ \textbf{power}')
plt.legend()
plt.tight_layout()
plt.show()

# plot time series
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, y)
plt.show()

result = period(
    t, y, dy,
    test_case['slope'], test_case['binning'],
    test_case['oversampling'], 10 * test_case['n_simulations'],
    test_case['percentile'],
)
