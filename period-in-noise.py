import numpy as np
from psresp import psresp
from sim_period import sim_period
from plot_period import plot_period


def spectrum(x, slope):
    y = x**slope

    return y


def timmerlc(slope, nt='None', dt='None', mean='None', sigma='None', seed='None'):
    # timmer alg from idl timmlc.pro
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

    simfreq = np.linspace(1, nt/2-1, num=nt/2, dtype='float64') / (dt*nt)
    simpsd = spectrum(simfreq, slope)
    fac = np.sqrt(simpsd)

    pos_real = np.random.RandomState(seed).normal(size=int(nt/2))*fac
    pos_imag = np.random.RandomState(seed).normal(size=int(nt/2))*fac

    pos_imag[int(nt/2)-1] = 0

    if float(nt/2.) > int(nt/2):
        neg_real = pos_real[0:int(nt/2)][::-1]
        neg_imag = -pos_real[0:int(nt/2)][::-1]
    else:
        neg_real = pos_real[0:int(nt/2)-1][::-1]
        neg_imag = -pos_real[0:int(nt/2)-1][::-1]

    real = np.hstack((0., pos_real, neg_real))
    imag = np.hstack((0., pos_imag, neg_imag))

    arg = real + 1j * imag
    rate = np.fft.ifft(arg).real
    time = dt*np.linspace(0, nt-1, nt, dtype='float')

    avg = np.mean(rate)
    std = np.sqrt(np.var(rate))

    rate = (rate - avg) * sigma / std + mean

    return dict(t=time, y=rate)


test_case = dict(slope=1.6, nt=1000, res=1, dy=np.ones([1000]),
         dt=np.array([2, 3, 4, 5, 6]), percentile=0.95, oversampling=10, n_simulations=100,
         df=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1]), slopes=np.linspace(1, 2.5, 16),
         binning=10, period=16, amplitude=.2
         )

test_data = timmerlc(-test_case['slope'], test_case['nt'], test_case['res'])

t = test_data['t'][300:701]
y = test_data['y'][300:701] + test_case['amplitude'] * np.sin(2 * np.pi * t / test_case['period'])
dy = test_case['dy'][300:701]

noise = psresp(
    t, y, dy,
    test_case['slopes'], test_case['dt'], test_case['df'], test_case['percentile'],
    test_case['oversampling'], test_case['n_simulations'],
    )

print(test_case['slope'], noise['slope'], noise['slope_error'])

nearest_slope = min(test_case['slopes'], key=lambda x:abs(x-noise['slope']))
print(nearest_slope)

obs_power, allpds, obs_nu = sim_period(t, y, dy,
                                       nearest_slope, test_case['binning'], 10*test_case['n_simulations'], test_case['oversampling'])

plot_period(obs_nu, obs_power, allpds, percentile=99)