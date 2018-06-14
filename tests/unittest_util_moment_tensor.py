#!/usr/bin/env python


import unittest
import numpy as np

from mtuq.util.moment_tensor.tape2015 import cmt2tt, cmt2tt15, tt2cmt, tt152cmt


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

        # perturb off-diagonal element ever so slightly
        #M[3] += np.random.normal(0., 1.e-6)
        M[4] += np.random.normal(0., 1.e-6)
        #M[5] += np.random.normal(0., 1.e-6)

        M1 = M
        gamma, delta, M0, kappa, theta, sigma = cmt2tt(M1)
        print 'gamma:', gamma
        print 'delta:', delta
        print 'M0:', M0
        print 'kappa:', kappa
        print 'theta:', theta
        print 'sigma:', sigma
        print '\n'
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
        print 'rho:', rho
        print 'v:', v
        print 'w:', w
        print 'kappa:', kappa
        print 'sigma:', sigma
        print 'h:', h

        M2 = tt152cmt(rho, v, w, kappa, sigma, h)
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





if __name__ == '__main__':
    unittest.main()



