
import warnings
import numpy as np



class Wavelet(object):
    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        raise NotImplementedError


    def arange(self, tmin, tmax, dt):
        """ Evaluates wavelet on the interval [tmin, tmax]
        """
        return self.trim(self.evaluate(np.arange(tmin, tmax, dt)))


    def linspace(self, tmin, tmax, nt):
        """ Evaluates wavelet on the interval [tmin, tmax]
        """
        return self.trim(self.evaluate(np.linspace(tmin, tmax, nt)))


    def trim(self, y):
        """ Trims zeros from beginning and end
        """
        # raise NotImplementedError
        return y


    def convolve_array(self, y, dt, mode=1):
        """ Convolves numpy array with given wavelet
        """
        warnings.warn('wavelet.convolve not yet implemented')
        return y

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
         trace.data = self.convolve_array(y, dt)
         return trace


    def convolve_stream(self, stream):
         """ Convolves obspy stream with given wavelet
         """
         for trace in stream:
             self.convolve_trace(trace)
         return stream



class Ricker(Wavelet):
    pass



class Gabor(Wavelet):
    pass



class DiracDelta(Wavelet):
    pass



