#!/usr/bin/env python


import unittest
import numpy as np

from mtuq.util.moment_tensor.tape2015 import cmt2tt, cmt2tt15, tt2cmt, tt152cmt


M = np.array([
    1.006279239004, # m11
    0.737173428960, # m22
    0.558314768020, # m33
   -0.231591759935, # m12
   -0.111675288138, # m13
    0.004991096805, # m23
    ])

EPSVAL = 1.e-6



class TestMomentTensor(unittest.TestCase):
    def test_2012(self):
        M1 = M
        # convert to 2012 parameters and back
        gamma, delta, M0, kappa, theta, sigma = cmt2tt(M1)
        M2 = tt2cmt(gamma, delta, M0, kappa, theta, sigma)

        e = np.linalg.norm(M1-M2)
        if e > EPSVAL:
            print '||M1 - M2|| = %e' % e
            raise Exception


    def test_2015(self):
        M1=M
        # convert to 2015 parameters and back
        rho, v, w, kappa, sigma, h = cmt2tt15(M1)
        M2 = tt152cmt(rho, v, w, kappa, sigma, h)

        e = np.linalg.norm(M1-M2)
        if e > EPSVAL:
            print '||M1 - M2|| = %e' % e
            raise Exception



if __name__ == '__main__':
    unittest.main()



