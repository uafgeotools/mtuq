#!/usr/bin/env python


import unittest
import numpy as np

from mtuq.wavelet import\
    Gaussian,\
    Triangle,\
    Trapezoid,\
    EarthquakeTrapezoid


EPSVAL = 1.e-3

def _is_close(a, b):
    if abs(a-b) < EPSVAL:
        return True
    else:
        print 'Error:', abs(a-b)
        return False


class TestWavelets(unittest.TestCase):

    def test_normalization_Gaussian(self):
        wavelet = Gaussian(sigma=1.)
        w = wavelet.evaluate_on_interval(half_duration=10., nt=10000)
        dt = (20./(10000-1))
        assert _is_close( dt*np.sum(w), 1. )


    def test_normalization_Triangle(self):
        wavelet = Triangle(half_duration=1.)
        w = wavelet.evaluate_on_interval(half_duration=1., nt=1000)
        dt = (2./(1000-1))
        assert _is_close( dt*np.sum(w), 1. )


    def test_normalization_Trapezoid(self):
        wavelet = Trapezoid(half_duration=1., rise_time=0.5)
        w = wavelet.evaluate_on_interval(half_duration=1., nt=1000)
        dt = (2./(1000-1))
        assert _is_close( dt*np.sum(w), 1. )


    def test_normalization_EarthquakeTrapezoid(self):
        wavelet = EarthquakeTrapezoid(rupture_time=0.5, rise_time=0.5)
        w = wavelet.evaluate_on_interval(half_duration=1., nt=1000)
        dt = (2./(1000-1))
        assert _is_close( dt*np.sum(w), 1. )



if __name__ == '__main__':
    unittest.main()


