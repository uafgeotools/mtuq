
import warnings
import numpy as np
from scipy import signal



class Wavelet(object):
    """ Base class for earthquake source-time functions and exploration
    geophysics source wavelets

    The ``evaluate`` method, which defines the mathematical expression for the
    the wavelet, must be implemented in a subclass

    When defining a particular wavelet through the ``evaluate`` method, an 
    analytical expression for the source as a function of time, a numerical 
    procedure, or a recorded time series can be used.

    For example, the following code defines a Guassian wavelet with unit 
    standard deviation:

    .. code::
        class Gaussian(Wavelet):
            def evaluate(self, t):
                return ((2*np.pi)**0.5)**(-1.)*np.exp(-0.5*(t)**2.)

    """

    def evaluate_on_interval(self, half_duration=None, nt=100):
        """ Evaluate wavelet on the interval
        """
        if not half_duration:
            raise ValueError
        assert half_duration > 0.

        t = np.linspace(-half_duration, +half_duration, nt)
        w = self.evaluate(t)

        if _is_symmetric(w):
            # trim symmetric wavelets only, otherwise any convolution results
            # will be off-center
            w = np.trim_zeros(w)

        return w


    def convolve_array(self, y, dt, mode=1):
        """ Convolves NumPy array with given wavelet
        """
        nt = len(y)
        half_duration = (nt-1)*dt/2.
        w = self.evaluate_on_interval(half_duration, nt)
        w *= dt

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
# user-supplied wavelet
#


class UserSupplied(Wavelet):
    """ Wavelet obtained from an arbitrary user-supplied time series
    """
    def __init__(self):
        raise NotImplementedError


#
# basic mathematical shapes and functions
#

class Triangle(Wavelet):
    def __init__(self, half_duration=None):
        if half_duration:
            self.half_duration = half_duration
        else:
            raise ValueError

    def evaluate(self, t):
        # construct a triangle with height = 1 and base = 2*half_duration
        w = 1. - abs(t)/self.half_duration
        w = np.clip(w, 0., np.inf)

        # area = (0.5)*(Wavelet)*(height)
        area = (0.5)*(2.*self.half_duration)*(1.)

        # normalize by area
        w /= area

        return w


class Trapezoid(Wavelet):
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


class Gaussian(Wavelet):
    def __init__(self, sigma=1., mu=0.):
        self.sigma = sigma
        self.mu = mu

    def evaluate(self, t):
        return ((2*np.pi)**0.5*self.sigma)**(-1.)*\
            np.exp(-0.5*((t-self.mu)/self.sigma)**2.)



#
# earthquake seismology "source-time functions" defined in terms of earthquake
# source parameters
#

def EarthquakeTrapezoid(rise_time=None, rupture_time=None):
    if not rise_time:
        raise ValueError

    if not rupture_time:
        raise ValueError

    assert rupture_time >= rise_time

    return Trapezoid(
        rise_time=rise_time,
        half_duration=(rise_time + rupture_time)/2.)




#
# exploration seismology "wavelets" defined in terms of dominant frequency
#

class GaussianWavelet(Wavelet):
    def __init__(self, dominant_frequency):
        raise NotImplementedError


class RickerWavelet(Wavelet):
    def __init__(self, dominant_frequency):
        # dominant frequency
        self.freq = dominant_frequency

    def evaluate(self, t):
        a = 2.*np.pi*self.freq
        return (1-0.5*(a*t)**2.)*np.exp(-0.25*(a*t)**2.)


class GaborWavelet(Wavelet):
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


