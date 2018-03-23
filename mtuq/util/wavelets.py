

import numpy as np



class Wavelet(object):
    def evaluate(self, t)
        """ Evaluates wavelet at chosen points
        """
        raise NotImplementedError


    def arange(self, tmin, tmax, dt):
        """ Evaluates wavelet on the interval [-tmax, +tmax]
        """
        return self.trim(self.evaluate(np.arange(tmin, tmax, dt)))


    def linspace(self, tmin, tmax, nt):
        """ Evaluates wavelet on the interval [-tmax, +tmax]
        """
        return self.trim(self.evaluate(np.linspace(tmin, tmax, nt)))


    def trim(self, y):
        """ Trims zeros from beginning and end
        """
        raise NotImplementedError


    def convolve(self, y, dt, mode=1):
         """ Convolves vector with given wavelet
         """
         nt = len(y)
         tmin = -(nt-1)*dt/2.
         tmax = +(nt-1)*dt/2.
         w = wavelet.arange(tmin, tmax, dt)

        if mode==1:
            # frequency-domain implementation
            return signal.fftconvolve(y, w, mode='same')

        elif mode==2:
            # time-domain implementation
            return np.convolve(y, w, mode='same')


    def convolve_trace(self, trace):
         """ Convolves obspy trace with given wavelet
         """
         try:
             y = trace.data
             dt = trace.stats.delta
         except:
             raise Exception

         return self.convolve(y, t)



class Trapezoid(Wavelet):
    """ Trapezoid-like wavelet obtained by convolving two boxes
        Reproduces capuaf:trap.c
    """

    def __init__(rise_time=None):
        if rise_time:
            self.rise_time = rise_time
        else:
            raise ValueError


    def evaluate(self, t):
        """ Evaluates wavelet at the chosen points
        """
        dt = 1000.

        # rather than an anlytical formula, the following numerical procedure
        # defines the trapezoid
        if t1>t2: t1,t2 = t2,t1
        n1 = max(int(round(t1/dt)),1)
        n2 = max(int(round(t2/dt)),1)
        r = 1./(n1+n2)
        y = np.zeros(n1+n2)
        for i in range(1,n1+1):
            y[i] = y[i-1] + r
            y[-i-1] = y[i]
        for i in range(i,n2):
            y[i] = y[i-1]

        # interpolate from numerical grid to the user-supplied points
        y = np.interp(t0,y0,t)



class Ricker(Wavelet):
    rasie NotImplementedError



class Gabor(Wavelet):
    rasie NotImplementedError



class DiracDelta(Wavelet):
    rasie NotImplementedError



