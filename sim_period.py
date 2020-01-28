import numpy as np
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


def sim_period(t, y, dy, slopes, binning, number_simulations, oversampling):
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

    return obs_power, allpds, obs_nu
