
import warnings
import numpy as np
from scipy import signal



class Base(object):
    """ Wavelet base class

       Provides methods for evaluating wavelets on intervals and convolving
       wavelets with user-supplied time series

       Specification of the wavelet itself is deferred to the subclass
    """

    def evaluate_on_interval(self, half_duration, nt):
        """ Evaluate wavelet on the interval
        """
        assert half_duration > 0.

        t = np.linspace(-half_duration, +half_duration, nt)
        w = self.evaluate(t)

        # trim symmetric wavelets only
        if _is_symmetric(w):
            w = np.trim_zeros(w)

        return w


    def convolve_array(self, y, dt, mode=1):
        """ Convolves numpy array with given wavelet
        """
        nt = len(y)
        half_duration = (nt-1)*dt/2.
        w = self.evaluate_on_interval(half_duration, nt)
        w /= np.sum(w)

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


    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        raise NotImplementedError("Must be implemented by subclass")


#
# basic mathematical shapes and functions
#

class Triangle(Base):
    def __init__(self, half_duration=None):
        if half_duration:
            self.half_duration = half_duration
        else:
            raise ValueError

    def evaluate(self, t):
        # construct a triangle with height = 1 and base = 2*half_duration
        w = 1. - abs(t)/half_duration
        w = np.max(w, 0.)

        # area = (0.5)*(base)*(height)
        area = (0.5)*(2.*half_duration)*(1.)

        # normalize by area
        w /= area

        return w


class Trapezoid(Base):
    """ Symmetric trapezoid
    """
    def __init__(self, rise_time=None, half_duration=None):
        if rise_time:
            self.rise_time = rise_time
        else:
            raise ValueError

        if half_duration:
            self.half_duration = half_duration
        else:
            raise ValueError

        assert rise_time <= half_duration


    def evaluate(self, t):
        # construct a trapezoid with height = 1
        w = (self.half_duration - abs(t))/self.rise_time
        w = np.clip(w, 0., 1.)


        top = 2*(self.half_duration-self.rise_time)
        bottom = 2.*self.half_duration
        height = 1.

        # normalize by area
        area = (0.5)*(top + bottom)*(height)
        w /= area

        return w


class Gaussian(Base):
    def __init__(self, sigma=1., mu=0.):
        self.sigma = sigma
        self.mu = mu

    def evaluate(self, t):
        a = (2*np.pi)**0.5*self.sigma
        return a**-1*np.exp(-(0.5*(t-self.mu)/self.sigma)**2.)



#
# earthquake seismology "source-time functions" defined in terms of earthquake
# source parameters
#

def EarthquakeTrapezoid(rise_time=None, rupture_time=None):
    if not rise_time:
        raise ValueError

    if not rupture_time:
        raise ValueError

    assert rupture_time > rise_time

    return Trapezoid(
        rise_time=rise_time,
        half_duration=(rise_time + rupture_time)/2.)




#
# exploration seismology "wavelets" defined in terms of dominant frequency
#

class GaussianWavelet(Base):
    def __init__():
        raise NotImplementedError


class RickerWavelet(Base):
    def __init__(self, dominant_frequency):
        # dominant frequency
        self.freq = dominant_frequency

    def evaluate(self, t):
        a = 2.*np.pi*self.freq
        return (1-0.5*(a*t)**2.)*np.exp(-0.25*(a*t)**2.)


class GaborWavelet(Base):
    def __init__(self, dominant_frequency):
        # dominant frequency
        self.freq = dominant_frequency

    def evaluate(self, t):
        a = np.pi*self.freq
        b = 2*np.pi*self.freq
        return np.exp(-(a*t)**2.)*np.cos(b*t)


#
# utility functions
#

def _is_symmetric(w):
    npts = len(w)

    if np.remainder(npts, 2) == 0:
        # even number of points
        return _is_close(w[:npts/2], w[npts/2:])

    else:
        # odd number of points
        return _is_close(w[:(npts-1)/2], w[(npts+1)/2:])


def _is_close(w1, w2):
    return np.all(np.isclose(w1, w2))


