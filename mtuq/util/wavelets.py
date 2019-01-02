
import warnings
import numpy as np
from scipy import signal



class Wavelet(object):
    """ Symmetric wavelet base class
    """

    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        raise NotImplementedError


    def evaluate_on_interval(self, half_duration, nt):
        """ Evaluate wavelet on the interval
        """
        assert half_duration > 0
        t = np.linspace(-half_duration, +half_duration, nt)
        y = self.evaluate(t)
        return np.trim_zeros(y)


    def convolve_array(self, y, dt, mode=1):
        """ Convolves numpy array with given wavelet
        """
        nt = len(y)
        half_duration = (nt-1)*dt/2.
        w = self.evaluate_on_interval(half_duration, nt)

        if mode==1:
            # frequency-domain implementation
            return signal.fftconvolve(y, w, mode='same')

        elif mode==2:
            # time-domain implementation
            return np.convolve(y, w, mode='same')


    def convolve(self, trace):
         """ Convolves obspy trace with given wavelet
         """
         try:
             y = trace.data
             dt = trace.stats.delta
         except:
             raise Exception
         trace.data = self.convolve_array(y, dt)
         return trace


class Trapezoid(Wavelet):
    """ Trapezoid-like wavelet obtained by convolving two boxes
        Reproduces capuaf:trap.c
    """

    def __init__(self, rise_time=None, rupture_time=None):
        if rise_time:
            self.rise_time = rise_time
        else:
            raise ValueError

        if rupture_time:
            self.rupture_time = rupture_time
        else:
            raise ValueError

        assert rupture_time > rise_time


    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        t1 = self.rise_time
        t2 = self.rupture_time
        dt = (t1+t2)/200

        n1 = max(int(round(t1/dt)),1)
        n2 = max(int(round(t2/dt)),1)

        t0 = np.linspace(-(t1+t2)/2., +(t1+t2)/2., n1+n2)
        y0 = np.zeros(n1+n2)

        r = 1./(n1+n2)
        for i in range(1,n1+1):
            y0[i] = y0[i-1] + r
            y0[-i-1] = y0[i]
        for i in range(i,n2):
            y0[i] = y0[i-1]

        y0 /= 0.5*np.sum(y0)
        y = np.interp(t,t0,y0)

        return y


class Triangle(Wavelet):
    def __init__(self, half_duration=None):
        if half_duration:
            self.half_duration = half_duration
        else:
            raise ValueError

    def evaluate(self, t):
        t0 = self.half_duration
        y = (1. - abs(t)/t0)
        return np.clip(y/t0, 0, np.inf)


class Gaussian(Wavelet):
    def __init__(self, sigma=1., mu=0.):
        self.sigma = sigma
        self.mu = mu

    def evaluate(self, t):
        return np.exp(-(0.5*(t-self.mu)/self.sigma)**2.)


class Ricker(Wavelet):
    def __init__(self, freq):
        # dominant frequency
        self.freq = freq

    def evaluate(self, t):
        a = 2.*np.pi*self.freq
        return (1-0.5*(a*t)**2.)*np.exp(-0.25*(a*t)**2.)


class Gabor(Wavelet):
    def __init__(self, freq):
        # dominant frequency
        self.freq = freq

    def evaluate(self, t):
        a = np.pi*self.freq
        b = 2*np.pi*self.freq
        return np.exp(-(a*t)**2.)*np.cos(b*t)
