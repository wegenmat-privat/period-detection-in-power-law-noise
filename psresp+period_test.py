import numpy as np

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


def _periodogram(series):
    series = series - np.mean(series)
    revff = np.fft.ifft(series)
    periodogram_bar = np.zeros(len(revff))
    freqs = np.zeros(len(revff))
    for freq, ff in enumerate(revff):
        if freq == 0:
            continue
        freqs[freq] = freq
        periodogram_bar[freq] = (abs(ff) ** 2)

    return freqs, periodogram_bar


def _rebinlc(time, rate, dt):
    # check for finite values
    rate = rate[np.isfinite(time)]
    time = time[np.isfinite(time)]

    # rebin lc/psd to evenly spaced time binning, from rebinlc.pro
    t = time

    ts = t[0]
    num = int((t[-1] - ts) / dt + 1)

    minnum = 1

    tn = np.empty([num])
    rn = np.empty([num])
    numperbin = np.empty([num])

    tlimit = ts + dt * np.linspace(0, num, num + 1)

    k = 0
    for i in range(num):
        tn[k] = tlimit[i]
        index = np.where(np.greater_equal(t, tlimit[i]) & (t < tlimit[i + 1]))
        number = len(t[index])

        numperbin[k] = number
        if np.greater_equal(number, minnum):
            rn[k] = np.sum(rate[index]) / number
            k = k + 1
        else:
            rn[k] = 0
            numperbin[k] = 0
            k = k + 1

    if k == 0:
        print('Rebinned lightcurve would not contain any data')

    if k != num:
        tn = tn[0:k]
        rn = rn[0:k]
        numperbin = numperbin[0:k]

    tn = tn[numperbin != 0]
    rn = rn[numperbin != 0]

    return tn, rn


def _do_periodogram(y):
    y = y - np.mean(y)
    freq, psd = _periodogram(y)

    return freq, psd


def _chi2_obs(norm, obs_pds, avg_pds, pds_err):
    obs_pds = norm * obs_pds
    a = (avg_pds - obs_pds) ** 2.
    b = pds_err ** 2.
    chi_obs = np.sum(a / b)

    return chi_obs


def _compare(obs_pds, avg_pds, pds_err, allpds, number_simulations):
    from scipy import optimize
    norm0 = 1.
    chi_obs = optimize.minimize(_chi2_obs, norm0, args=(obs_pds, avg_pds, pds_err), method='SLSQP').fun

    chi_dist = np.empty([number_simulations])
    for n in range(number_simulations):
        a = (allpds[n, :] - avg_pds) ** 2
        b = pds_err ** 2
        chi = a / b
        chi_dist[n] = np.sum(chi)

    suf = 0
    for i in range(len(chi_dist)):
        if np.greater_equal(chi_dist[i], chi_obs):
            suf = suf + 1

    suf = suf / len(chi_dist)

    return suf


def _psresp_pro(t, y, dy, slopes, number_simulations, binning, oversampling, df):

    bin = binning
    date, rat, raterr = t, y, dy
    date = date - date[0]
    duration = np.max(date) - np.min(date)
    npoints = int(duration / bin) * number_simulations * oversampling
    params = -slopes
    lc_variance = np.var(rat) - np.var(raterr)

    # observed PDS calculation
    obs_nu, obs_pds = _do_periodogram(rat)
    obs_nu = np.log10(obs_nu)

    # normalisation
    obs_pds = (2. * duration) / (np.mean(rat) * np.mean(rat) * len(rat) * len(rat)) * obs_pds

    # rebin
    obs_freqs, obs_power = _rebinlc(obs_nu, obs_pds, dt=df)
    obs_power = np.log10(obs_power)

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
        # print('computing PDS ' + str(j+1))

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

        # calculate PDS of simulated light curve
        sim_nu, sim_pds = _do_periodogram(binrate)
        sim_nu = np.log10(sim_nu)
        # sim_pds = sim_pds + p_noise + p_alias

        # normalisation
        sim_pds = (2. * (np.max(bintime) - np.min(bintime))) / (np.mean(binrate)
                                                                * np.mean(binrate) * len(binrate) * len(
            binrate)) * sim_pds

        # rebin simulated PDS in same manner as observed
        logfreqs, power = _rebinlc(sim_nu, sim_pds, dt=df)
        logpower = np.log10(power)

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


def psresp(t, y, dy, slopes, dt, df, percentile, oversampling=10, number_simulations=100):
    """
    Compute power spectral density of a light curve assuming an unbroken power law with the PSRESP method.
    The artificial light curves are generated using the algorithm by Timmer and Koenig (1995).
    For an introduction to the PSRESP method, see Uttley (2002).
    The function returns a results dictionary with the following content:
    - ``slope`` (`float`) -- Mean slope of the power law
    - ``slope_error`` (`float`) -- Error of the slope of the power law
    - ``suf`` (`~numpy.ndarray`) -- Success fraction for each model parameter
    - ``best_parameters`` (`~numpy.ndarray`) -- Parameters satisfying the significance criterion
    - ``statistics`` (`~numpy.ndarray`) -- Data used to calculate the mean slope and its error
      over a grid of ``dt`` and ``df``
        - slope with the highest success fraction
        - highest success fraction
        - slope of the lower full width at half maximum for the success fraction distribution
        - slope of the higher full width at half maximum for the success fraction distribution
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
    dt : `~numpy.ndarray`
        bin length for the light curve in units of ``t``
    df : `~numpy.ndarray`
        bin factor for the logarithmic periodogram
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

    # import matplotlib.pyplot as plt
    # from matplotlib import gridspec
    # from matplotlib import rc
    # fig = plt.figure(figsize=(4, 4))
    # rc('text', usetex=True)
    # rc('font', size=18)
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # gs = gridspec.GridSpec(1, 1)
    # ax1 = fig.add_subplot(gs[0, 0])

    # ax1.plot(t, y)

    from scipy import interpolate
    t_ini, y_ini, dy_ini = t, y, dy
    suf = np.empty([len(slopes), len(dt), len(df)])
    statistics = np.empty([4, len(dt), len(df)])
    for b in range(len(dt)):
        # print('binning: ' + str(dt[b]))
        t, y = _rebinlc(t_ini, y_ini, dt[b])
        t, dy = _rebinlc(t_ini, dy_ini, dt[b])

        for f in range(len(df)):
            # print('df: ' + str(df[f]))
            for s in range(len(slopes)):
                # print('slope: ' + str(slopes[s]))

                # psresp
                faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds, sim_pds, obs_nu = \
                    _psresp_pro(t, y, dy, slopes[s], number_simulations, dt[b], oversampling, df[f])

                # do chi2
                suf[s, b, f] = _compare(obs_power, avg_pds, pds_err, allpds, number_simulations)

            # find best slope and estimate error
            best_slope = slopes[np.argmax(suf[:, b, f])]
            best_slope_suf = np.max(suf[:, b, f])

            slopes_fwhm = interpolate.UnivariateSpline(slopes, suf[:, b, f] - 0.5 * best_slope_suf, s=0).roots()
            if len(slopes_fwhm) == 0:
                slopes_fwhm = [np.nan]
            low_slopes = slopes_fwhm[0]
            high_slopes = slopes_fwhm[-1]
            if low_slopes == high_slopes:
                low_slopes = high_slopes = np.nan

            statistics[0, b, f] = best_slope
            statistics[1, b, f] = best_slope_suf
            statistics[2, b, f] = low_slopes
            statistics[3, b, f] = high_slopes

    # ax1.plot(faketime, fakerate)
    # ax1.plot(bintime, binrate)
    # for i in range(number_simulations):
    #     ax1.plot(1. / obs_freqs, allpds[i, :], c='silver', alpha=0.5)
    # ax1.plot(1. / obs_freqs, obs_power)
    # ax1.plot(1. / obs_freqs, allpds[14, :], 'g')
    # ax1.plot(1. / obs_freqs, allpds[55, :], 'g')
    # ax1.plot(1. / obs_freqs, allpds[15,:], 'r')
    # ax1.plot(1. / obs_freqs, allpds[56, :], 'r')
    # ax1.plot(1. / obs_freqs, avg_pds, 'k')
    # ax1.plot(1. / obs_nu, np.log10(sim_pds))
    # ax1.set(xlabel=r'$\lg$ \textbf{periods} (d)', ylabel=r'$\lg$ \textbf{power}')
    # ax1.set(xlabel=r'\textbf{time} (d)', ylabel=r'\textbf{flux} (a.u.)')
    # plt.tight_layout()
    # plt.show()

    test_significance = np.percentile(statistics[1, :, :], 95 * percentile)
    statistics_test = (np.greater_equal(statistics[1, :, :], test_significance)) & \
                      (np.isfinite(statistics[2, :, :])) & \
                      (np.isfinite(statistics[3, :, :]))
    # statistics_test = statistics[2, :, :] + statistics[3, :, :] == np.min(statistics[2, :, :] + statistics[3, :, :])
    best_parameters = np.where(statistics_test == True)

    mean_slope = np.sum(statistics[0, :, :][statistics_test]
                        * statistics[1, :, :][statistics_test])\
                        / (np.sum(statistics[1, :, :][statistics_test]))
    slope_error = np.abs(np.min(statistics[2, :, :][statistics_test]) - np.max(statistics[3, :, :][statistics_test]))

    return dict(slope=mean_slope,
                slope_error=slope_error,
                suf=suf,
                best_parameters=best_parameters,
                statistics=statistics
                )


def plot_psresp(slopes, dt, df, suf, mean_slope, slope_error, best_parameters, statistics):
    """
    Plot the success fraction over slopes for parameters satisfying the significance criterion
    and the histogram over the grid of parameters.
    Parameters
    ----------
    slopes : `~numpy.ndarray`
        slopes of the power law model
    dt : `~numpy.ndarray`
        bin length for the light curve in units of ``t``
    df : `~numpy.ndarray`
        bin factor for the logarithmic periodogram
    suf : `~numpy.ndarray`
        Success fraction for each model parameter
    mean_slope : `~float`
        Mean slope of the power law
    slope_error : `~float`
        Error of the mean slope
    best_parameters : `~numpy.ndarray`
        Parameters satisfying the significance criterion
    statistics : `~numpy.ndarray`
        Data used to calculate the mean slope and its error
        over a grid of ``dt`` and ``df``
        - slope with the highest success fraction
        - highest success fraction
        - slope of the lower full width at half maximum for the success fraction distribution
        - slope of the higher full width at half maximum for the success fraction distribution
    Returns
    -------
    fig : `~matplotlib.Figure`
        Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    m = (-1 / (-np.max(slopes) / np.min(slopes) + 1)) / np.min(slopes)
    n = 1 / (-np.max(slopes) / np.min(slopes) + 1)

    # fig = plt.figure(figsize=(12, 8))
    fig = plt.figure(figsize=(8, 4))
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('font', size=18)
    for indx in range(len(best_parameters[0])):
        plt.plot(slopes, suf[:, dt == dt[best_parameters[0][indx]], df == df[best_parameters[1][indx]]],
                 label=r'$\Delta t_{{bin}}$ = {}, $\Delta f$ = {}'.format(dt[best_parameters[0][indx]],
                                                                          df[best_parameters[1][indx]])
                 )
    plt.axhline(y=0.5 * np.max(statistics[1, :, :][best_parameters]),
                xmin=np.min(statistics[2, :, :][best_parameters]) * m + n,
                xmax=np.max(statistics[3, :, :][best_parameters]) * m + n,
                color='k'
                )
    plt.axvline(x=np.round(mean_slope, 1),
                ymin=0,
                ymax=np.max(statistics[1, :, :][best_parameters]),
                color='k'
                )
    plt.text(mean_slope - 0.1501,
             0.5 * np.max(statistics[1, :, :][best_parameters]) + 0.01,
             'FWHM = {:.1f}'.format(slope_error))
    plt.text(mean_slope + 0.15, np.max(statistics[1, :, :][best_parameters]) - 0.1, 'mean slope = {:.1f}'.format(mean_slope))
    plt.xticks(np.arange(np.min(slopes), np.max(slopes), .2))
    plt.xlabel(r'\textbf{slope}')
    plt.ylabel(r'\textbf{success fraction}')
    # plt.xlim(np.min(slopes), np.max(slopes))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot contour
    # get test data
    X, Y, Z = slopes, dt, df
    suf_test = np.greater_equal(suf, np.min(statistics[1, :, :][best_parameters]))
    suf_test = np.zeros_like(suf, bool)
    suf_test[np.where(suf[:, best_parameters[0], best_parameters[1]] == np.max(suf[:, best_parameters[0], best_parameters[1]], axis=0))[0], best_parameters[0], best_parameters[1]] = True
    sumx = np.sum(suf_test, axis=0)
    sumy = np.sum(suf_test, axis=1)
    sumz = np.sum(suf_test, axis=2)
    max_sumx = np.max(sumx)
    max_sumy = np.max(sumy)
    max_sumz = np.max(sumz)
    max_colours = np.max([max_sumx, max_sumy, max_sumz])

    # set up axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    # set up colormap
    base = plt.cm.get_cmap('hot_r')
    if max_colours == 1:
        color_list = base(np.linspace(1 / 2, 1, 2))
        cmap_name = base.name + str(2)
        cmap = base.from_list(cmap_name, color_list, 2)
    else:
        color_list = base(np.linspace(1 / max_colours, 1, max_colours))
        cmap_name = base.name + str(max_colours)
        cmap = base.from_list(cmap_name, color_list, max_colours)
    cmap.set_under(color='white', alpha=0)

    # plot contour
    y, z = np.meshgrid(Y, Z)
    yz = ax.contourf(sumx.T, y, z, zdir='x', offset=np.min(slopes), cmap=cmap, vmin=.5, vmax=max_colours + .5)

    x, z = np.meshgrid(X, Z)
    xz = ax.contourf(x, sumy.T, z, zdir='y', offset=np.max(dt), cmap=cmap, vmin=.5, vmax=max_colours + .5)

    x, y = np.meshgrid(X, Y)
    xy = ax.contourf(x, y, sumz.T, zdir='z', offset=np.min(df), cmap=cmap, vmin=.5, vmax=max_colours + .5)

    # set axes limits
    ax.set_xlim([np.min(slopes), np.max(slopes)])
    ax.set_ylim([np.min(dt), np.max(dt)])
    ax.set_zlim([np.min(df), np.max(df)])

    # set axes labels
    ax.set_xlabel('slope', labelpad=10)
    ax.set_xticks(np.arange(np.min(slopes), np.max(slopes), .2))
    ax.set_ylabel(r'$\Delta t_{bin}$', labelpad=10)
    ax.set_yticks(dt)
    ax.set_zlabel(r'$\Delta f$', labelpad=10)
    ax.set_zticks(df)

    # plot colorbar
    cyz = plt.colorbar(yz, ticks=np.arange(max_sumx + 1))
    cxz = plt.colorbar(xz, ticks=np.arange(max_sumy + 1))
    cxy = plt.colorbar(xy, ticks=np.arange(max_sumz + 1))

    # set colorbar labels
    cyz.ax.set_title(r'$\Delta t_{{bin}}$/' + '\n' r'$\Delta f$')
    cxy.ax.set_title('slope/\n' + r'$\Delta t_{bin}$')
    cxz.ax.set_title('slope/\n' + r'$\Delta f$')

    plt.tight_layout()
    plt.show()


def _psresp_period(t, y, dy, slopes, binning, number_simulations, oversampling):
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
        print('computing PDS ' + str(j+1))

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


def period(t, y, dy, slopes, binning, oversampling=10, number_simulations=100):
    """
    Compute power spectral density of a light curve assuming an unbroken power law with the PSRESP method.
    The artificial light curves are generated using the algorithm by Timmer and Koenig (1995).
    For an introduction to the PSRESP method, see Uttley (2002).
    The function returns a results dictionary with the following content:
    - ``slope`` (`float`) -- Mean slope of the power law
    - ``slope_error`` (`float`) -- Error of the slope of the power law
    - ``suf`` (`~numpy.ndarray`) -- Success fraction for each model parameter
    - ``best_parameters`` (`~numpy.ndarray`) -- Parameters satisfying the significance criterion
    - ``statistics`` (`~numpy.ndarray`) -- Data used to calculate the mean slope and its error
      over a grid of ``dt`` and ``df``
        - slope with the highest success fraction
        - highest success fraction
        - slope of the lower full width at half maximum for the success fraction distribution
        - slope of the higher full width at half maximum for the success fraction distribution
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
    dt : `~numpy.ndarray`
        bin length for the light curve in units of ``t``
    df : `~numpy.ndarray`
        bin factor for the logarithmic periodogram
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
    t_ini, y_ini, dy_ini = t, y, dy
    dt = 1
    # for s in range(len(slopes)):
    #     print('slope: {}, dt: {}, df: {}'.format(slopes[s], dt, df[s]))
    #     t, y = _rebinlc(t_ini, y_ini, dt)
    #     t, dy = _rebinlc(t_ini, dy_ini, dt)

    # psresp
    print('begin psresp')
    faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds, sim_pds, obs_nu = \
                _psresp_period(t, y, dy, slopes, binning, number_simulations, oversampling)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(1. / obs_nu, obs_power, label=r'obs')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 99, axis=0), label=r'99\%-sim')
    ax1.plot(1. / obs_nu, np.percentile(allpds, 95, axis=0), label=r'95\%-sim')
    # ax1.plot(1. / obs_nu, np.max(allpds, axis=0), label='max-sim')
    print(1. / obs_nu[np.where(obs_power>np.percentile(allpds, 99, axis=0))])
    plt.xlabel(r'\textbf{periods} (d)')
    plt.ylabel(r'$ \mathcal{{LSP}} $ \textbf{power}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return dict(periods=1. / obs_nu[np.where(obs_power>np.percentile(allpds, 99, axis=0))])

print('go')

nt = 1000
subset = 400
binning = 1
n_simulations = 100
periods = np.empty([n_simulations, 2])
period_p = []

test_case = dict(slope=1.6, nt=nt, res=1, dy=np.ones([nt]),
                     dt=np.array([2, 3, 4, 5, 6]), percentile=0.95, oversampling=10, n_simulations=100,
                     df=np.array([0.3, 0.5, 0.7, 0.9, 1.1]), slopes= np.linspace(0.5, 2.5, 21)
                     )


t_ini, y_ini, = _timmerlc(-test_case['slope'], test_case['nt'], test_case['res'])
y_ini = y_ini + .2 * np.sin(2 * np.pi * t_ini / 16 )

print('test set ready')

for fraction in range(1):
     observation = np.random.RandomState(42).choice(np.arange(0, subset, 1), int(subset/(fraction + 1)), replace=False)
     t, y, dy = t_ini[int((nt-subset)/2):int((nt+subset)/2)+1][np.sort(observation)],\
                y_ini[int((nt-subset)/2):int((nt+subset)/2)+1][np.sort(observation)],\
                test_case['dy'][int((nt-subset)/2):int((nt+subset)/2)+1][np.sort(observation)]
#
#     # result = psresp(
#     #     t, y, dy,
#     #     test_case['slopes'], test_case['dt'], test_case['df'], test_case['percentile'],
#     #     test_case['oversampling'], test_case['n_simulations'],
#     # )
#     # print(result['slope'])
#     #
#     # slopes = result['statistics'][0, :, :][result['best_parameters']]
#     # sufs = result['statistics'][1, :, :][result['best_parameters']]
#     # dt = test_case['dt'][result['best_parameters'][0]]
#     # df = test_case['df'][result['best_parameters'][1]]
#     #
#     # print(slopes, sufs, dt, df)
#     #
#     # plot_psresp(
#     #         test_case['slopes'], test_case['dt'], test_case['df'], result['suf'],
#     #         result['slope'], result['slope_error'], result['best_parameters'], result['statistics'],
#     # )

# test on LombScargle
from astropy.stats import LombScargle
obs_nu = 1. / np.arange(1, 100 + 1, 1)
obs_pds = LombScargle(t, y, dy).power(obs_nu)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
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
# t, y, dy = t_ini, y_ini, test_case['dy']
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(1, 1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(t,y)
# plt.show()
#
# from astropy.stats import LombScargle
# obs_nu = 1. / np.arange(1, 100 + 1, 1)
# obs_pds = LombScargle(t, y, dy).power(obs_nu)
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(1, 1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(1. / obs_nu, obs_pds)
# plt.show()

result = period(
    t, y, dy,
    1.6, binning,
    test_case['oversampling'], 10*test_case['n_simulations'],
)

#    periods[fraction, 0] = 1 / (fraction + 1)
#    periods[fraction, 1] = len(result['periods'])
#    period_p.append(result['periods'])

# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', size=18)
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(1, 1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(periods[:, 0], periods[:, 1])
# plt.tight_layout()
# plt.show()
#
# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(1, 1)
# ax2 = fig.add_subplot(gs[0, 0])
# cmap = plt.get_cmap('hot_r', 4)
# cmap.set_over('black')
# bins = np.linspace(1, 100, 100)
# hist = np.empty([len(bins)-1, n_simulations])
# for s in range(n_simulations):
#     hist[:, s], bin_edges = np.histogram(period_p[s], bins=bins)
# X, Y = np.meshgrid(bin_edges[0:-1], np.linspace(0, 1, n_simulations+1))
# Z = hist.T
# cp = ax2.pcolormesh(X, Y, Z, cmap=cmap)
# plt.tight_layout()
# plt.show()
