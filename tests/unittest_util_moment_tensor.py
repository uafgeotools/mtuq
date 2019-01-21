#!/usr/bin/env python


import unittest
import numpy as np

from mtuq.util.moment_tensor.tape2015 import cmt2tt, cmt2tt15, tt2cmt, tt152cmt
from mtuq.util.moment_tensor.basis import change_basis
from mtuq.util.math import PI, DEG


EPSVAL = 1.e-6



class TestMomentTensor(unittest.TestCase):
    def test_FullMomentTensor_2012(self):
        M1 = np.array([
            1.006279239004, # m11
            0.737173428960, # m22
            0.558314768020, # m33
           -0.231591759935, # m12
           -0.111675288138, # m13
            0.004991096805, # m23
            ])

        # convert to 2012 parameters and back
        gamma, delta, M0, kappa, theta, sigma = cmt2tt(M1)
        M2 = tt2cmt(gamma, delta, M0, kappa, theta, sigma)

        e = np.linalg.norm(M1-M2)
        if e > EPSVAL:
            print '||M1 - M2|| = %e' % e
            raise Exception


    def test_FullMomentTensor_Tape2015(self):
        M1 = np.array([
            1.006279239004, # m11
            0.737173428960, # m22
            0.558314768020, # m33
           -0.231591759935, # m12
           -0.111675288138, # m13
            0.004991096805, # m23
            ])

        # convert to 2015 parameters and back
        rho, v, w, kappa, sigma, h = cmt2tt15(M1)
        M2 = tt152cmt(rho, v, w, kappa, sigma, h)

        e = np.linalg.norm(M1-M2)
        if e > EPSVAL:
            print '||M1 - M2|| = %e' % e
            raise Exception


    def test_RandomFullMomentTensor_2012(self):
        pass


    def test_RandomFullMomentTensor_Tape2015(self):
        pass


    def test_Explosion_2012(self):
        M = np.array([
            1., # m11
            1., # m22
            1., # m33
            0., # m12
            0., # m13
            0., # m23
            ])

        # perturb off-diagonal element slightly
        M[3] += np.random.normal(0., 1.e-6)
        #M[4] += np.random.normal(0., 1.e-6)
        #M[5] += np.random.normal(0., 1.e-6)

        M1 = M
        gamma, delta, M0, kappa, theta, sigma = cmt2tt(M1)
        assert abs(delta - 90.) < 1.e-3

        M2 = tt2cmt(gamma, delta, M0, kappa, theta, sigma)
        e = np.linalg.norm(M1-M2)
        if e > 1.e-3:
            print '||M1 - M2|| = %e' % e
            raise Exception


    def test_Explosion_Tape2015(self):
        M = np.array([
            1., # m11
            1., # m22
            1., # m33
            0., # m12
            0., # m13
            0., # m23
            ])

        # perturb off-diagonal elements slightly
        M[3] += np.random.normal(0., 1.e-6)
        #M[4] += np.random.normal(0., 1.e-6)
        #M[5] += np.random.normal(0., 1.e-6)

        M1 = M
        rho, v, w, kappa, sigma, h = cmt2tt15(M1)
        M2 = tt152cmt(rho, v, w, kappa, sigma, h)
        e = np.linalg.norm(M1-M2)
        if e > 1.e-3:
            print '||M1 - M2|| = %e' % e
            raise Exception


    def test_Tape2015_appendixA(self):
        """ Reproduces the calculations in Appendix A of
            Tape and Tape (2015), "A uniform parameterization of moment tensors"
        """
        rho = 1.

        # Tape2015 parameters given in the appendix
        u = 3.*PI/8.
        v = -1./9.
        kappa = 4.*PI/5. * DEG
        sigma = -PI/2. * DEG
        h = 3./4.

        # moment tensor given in the appendix
        mt0 = np.array([
            0.196, # m11
            0.455, # m22
           -0.651, # m33
           -0.397, # m12
           -0.052, # m13
            0.071, # m23
            ])

        # w is like lune latitude delta, whereas u is like lune colatitude beta
        w = 3.*PI/8. - u

        mt = tt152cmt(rho, v, w, kappa, sigma, h)

        # convert from up-south-east to north-west-up
        mt = change_basis(mt, 1, 3)

        e = np.linalg.norm(mt-mt0)
        if e > 1.e-3:
            print '||M1 - M2|| = %e' % e
            raise Exception



if __name__ == '__main__':
    unittest.main()



