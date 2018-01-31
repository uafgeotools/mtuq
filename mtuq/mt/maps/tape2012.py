#!/usr/bin/env python

EPSVAL = 1e-6
VERBOSE = 0


import numpy as np

from mtuq.mt.change_basis import change_basis
from mtuq.util.math import PI, DEG, eig, fangle_signed, rotmat, rotmat_gen, wrap360


bounds = ({
    'v': [vmin, vmax],
    'w': [wmin, wmax],
    'h': [hmin, hmax],
    'delta': [dmin, dmax],
    'theta': [tmin, tmax],
    'kappa': [kmin, kmax],
    })



def forward(M):
    """
    Converts up-south-east moment tensor to 2012 parameters

    input: M: moment tensor with shape [6]
              must be in up-south-east (GCMT) convention

    output: gamma, delta, M0, kappa, theta, sigma
    """
    # diagonalize
    lam, U, = eig(_mat(_change_basis(M)), sort_type=1)
    gamma, delta, M0, = lam2lune(lam)

    # require det(U) = 1
    U = _fixdet(U)

    Y = rotmat(45,1)
    V = np.dot(U, Y)

    S = V[:,0] # slip vector
    N = V[:,2] # fault normal

    # fix roundoff
    N = _round0(N); S = _round0(S)
    N = _round1(N); S = _round1(S)

    # find the angles corresponding to the bounding region shown in
    # TT2012, Figs.16,B1
    theta, sigma, kappa, _ = frame2angles(N,S)

    return (
        gamma,
        delta,
        M0,
        kappa,
        theta,
        sigma)


def inverse(M):
    """
    Converts up-south-east moment tensor to 2015 parameters

    input: M: moment tensor with shape [6]
              must be in up-south-east (GCMT) convention

    output: kappa, sigma, M0, v, w, h
    """
    gamma, delta, M0, kappa, theta, sigma = cmt2tt(M)
    rho = np.sqrt(2.)*M0
    v, w = lune2rect(gamma, delta)
    h = np.cos(theta/DEG)

    return (
        rho,
        v,
        w,
        kappa,
        sigma,
        h)


def tt2cmt(gamma, delta, M0, kappa, theta, sigma):
    """
    Converts 2012 parameters to up-south-east moment tensor

    input: gamma, delta, M0, kappa, theta, sigma

    output: M: moment tensor with shape [6]
               in up-south-east (GCMT) convention
    """
    lam = lune2lam(gamma, delta, M0)

    # TT2012, p.485
    phi = -kappa

    north = np.array([-1, 0, 0])
    zenith = np.array([0, 0, 1])

    K = np.dot(rotmat(phi, 2), north)
    N = np.dot(rotmat_gen(K, theta), zenith)
    S = np.dot(rotmat_gen(N, sigma), K)

    # TT2012, eq.28
    Y = rotmat(-45,1)

    V = np.column_stack([S, np.cross(N,S), N])
    U = np.dot(V, Y)
    M = np.dot(np.dot(
            U, 
            np.diag(lam)),
            U.T)

    # convert from south-east-up to up-south-east convention
    # (note: U is still in south-east-up)
    M = change_basis(_vec(M), 5, 1)

    return M


def tt152cmt(rho, v, w, kappa, sigma, h):
    """
    Converts 2015 parameters to up-south-east moment tensor

    input: kappa, sigma, M0, v, w, h

    output: M: moment tensor with shape [6]
               in up-south-east (GCMT) convention
    """
    theta = np.arccos(h)*DEG
    M0 = rho/np.sqrt(2)
    gamma, delta = rect2lune(v, w)
    M = tt2cmt(gamma, delta, M0, kappa, theta, sigma)
    return M



### eigenvalue-related functions
    

def lam2lune(lam):
    """
    Converts moment tensor eigenvalues to lune coordinates

    input
    : lam: vector with shape [3]

    output
    : gamma: angle from DC meridian to lune point (-30 <= gamma <= 30)
    : delta: angle from deviatoric plane to lune point (-90 <= delta <= 90)
    : M0: seismic moment, M0 = ||lam|| / sqrt(2)
    """
    # descending sort
    lam = np.sort(lam)[::-1]

    # magnitude of lambda vector (rho of TapeTape2012a p.490)
    lammag = np.linalg.norm(lam)

    # seismic moment
    M0 = lammag/np.sqrt(2.)

    # TapeTape2012a, eqs.21a,23
    # numerical safety 1: if trace(M) = 0, delta = 0
    # numerical safety 2: is abs(bdot) > 1, adjust bdot to +1 or -1
    if np.sum(lam) != 0.:
        bdot = np.sum(lam)/(np.sqrt(3)*lammag)
        np.clip(bdot, -1, 1)
        delta = 90. - np.arccos(bdot)*DEG
    else:
        delta = 0.
    
    # TapeTape2012a, eq.21a
    # note: we set gamma=0 for (1,1,1) and (-1,-1,-1)
    if lam[0] != lam[2]:
        gamma = np.arctan((-lam[0] + 2.*lam[1] - lam[2])
                         /(np.sqrt(3)*(lam[0] - lam[2]))) * DEG
    else:
        gamma = 0.

    return (
        gamma,
        delta,
        M0,
        )


def lune2lam(gamma, delta, M0):
    """ Converts lune coordinates to moment tensor eigenvalues
    """
    beta = 90. - delta

    # magnitude of lambda vectors (TT2012, p.490)
    rho = M0*np.sqrt(2)

    # convert to eigenvalues (TT2012, Eq.20)
    # matrix to rotate points such that delta = 90 is (1,1,1) and delta = -90 is (-1,-1,-1)
    R = np.array([[3.**0.5, 0., -3.**0.5],
                  [-1., 2., -1.],
                  [2.**0.5, 2.**0.5, 2.**0.5]])/6.**0.5

    # Cartesian points as 3 x n unit vectors (TT2012, Eq.20)
    #Pxyz = latlon2xyz(delta,gamma,ones(n,1))
    Pxyz = np.array([np.cos(gamma/DEG)*np.sin(beta/DEG),
                     np.sin(gamma/DEG)*np.sin(beta/DEG),
                     np.cos(beta/DEG)])

    # rotate points and apply magnitudes
    lamhat = np.dot(R.T, Pxyz)
    lam = rho*lamhat

    return rho*lamhat


def lune2rect(gamma, delta):
    """
    Converts eigenvalues to lune coordinates

    : type gamma: float
    : type delta: float
    : return: (v, w)
    """
    # convert to radians
    delta /= DEG
    gamma /= DEG
    beta = PI/2. - delta

    v = gamma2v(gamma)
    u = beta2u(beta)
    w = 3.*PI/8. - u

    return v, w


def rect2lune(v, w):
    u = 3.*PI/8. - w

    gamma = v2gamma(v)
    beta = u2beta(u)

    # convert to degrees
    gamma *= DEG
    beta *= DEG 
    delta = 90 - beta

    return gamma, delta


def beta2u(beta):
    """ See eq ? TapeTape2015
    """
    u = (0.75*beta 
          - 0.5*np.sin(2.*beta)
          + 0.0625*np.sin(4.*beta))
    return u


def gamma2v(gamma):
    """ See eq ? TT2015
    """
    v = (1./3.)*np.sin(3.*gamma)
    return v


def u2beta(u, N=1000):
    """ See eq ? TT2015
    """
    beta0 = np.linspace(0, PI, N)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(u,u0,beta0)
    return beta


def v2gamma(v):
    """ See eq ? TT2015
    """
    return (1./3.)*np.arcsin(3.*v)



### eigenvector-related functions


def faultvec2angles(S,N):
    """ Returns fault angles in degrees,
        assumes input vectors in south-east-up basis
    """

    # for north-west-up basis (as in TT2012)
    #zenith = [0 0 1]'; north  = [1 0 0]';

    # for up-south-east basis (GCMT)
    #zenith = [1 0 0]'; north  = [0 -1 0]';

    # for south-east-up basis (as in TT2012)
    #zenith = [0 0 1]'; north  = [-1 0 0]';

    zenith = np.array([0, 0, 1])
    north  = np.array([-1, 0, 0])

    # strike vector from TT2012, Eq. 29
    v = np.cross(zenith,N)
    if np.linalg.norm(v)==0:
        # TT2012 Appendix B
        if VERBOSE > 0:
            print('horizontal fault -- strike vector is same as slip vector')
        K = S
    else:
        K = v / np.linalg.norm(v)

    # Figure 14
    kappa = fangle_signed(north,K,-zenith)

    # Figure 14
    costh = np.dot(N,zenith)
    theta = np.arccos(costh)*DEG

    # Figure 14
    sigma = fangle_signed(K,S,N)

    kappa = wrap360(kappa)

    return (theta,sigma,kappa,K,)


def frame2angles(N,S):
    """
     There are four combinations of N and S that represent a double couple
     moment tensor, as shown in Figure 15 of TT2012.
     From these four combinations, there are two possible fault planes.
     We want to isolate the combination that is within the bounding
     region shown in Figures 16 and B1.
    """
    # four combinations for a given frame
    S1 =  S; N1 =  N
    S2 = -S; N2 = -N
    S3 =  N; N3 =  S
    S4 = -N; N4 = -S

    # calculate fault angles for each combination
    (theta1,sigma1,kappa1,K1,) = faultvec2angles(S1,N1)
    (theta2,sigma2,kappa2,K2,) = faultvec2angles(S2,N2)
    (theta3,sigma3,kappa3,K3,) = faultvec2angles(S3,N3)
    (theta4,sigma4,kappa4,K4,) = faultvec2angles(S4,N4)

    theta = np.array([theta1, theta2, theta3, theta4])
    sigma = np.array([sigma1, sigma2, sigma3, sigma4])
    kappa = np.array([kappa1, kappa2, kappa3, kappa4])
    K = np.array([K1, K2, K3, K4])

    # which combination lies within the bounding region?
    btheta = (theta <= 90.+EPSVAL)
    bsigma = (abs(sigma) <= 90.+EPSVAL)
    bb = np.logical_and(btheta, bsigma)
    ii = np.where(bb)[0]
    nn = len(ii)

    if nn==0:
        raise Exception('no match')
    elif nn==1:
        jj = ii[0]
    elif nn==2:
        # choose one of the two
        jj = _pick(ii,theta,sigma,kappa)
        if VERBOSE > 0:
            print('moment tensor on boundary (#d candidates)' % length(ii))
    else:
        # just take the first one in the list, for now
        # this is a more unusual case, like for horizontal faults
        jj = ii[0]
        if VERBOSE > 0:
            print('moment tensor on boundary of orientation domain (#d candidates)' %
                  length(ii))

    return (theta[jj], sigma[jj], kappa[jj], K[jj],)


def _pick(idx,theta,sigma,kappa):
    """
    Choose between two moment tensor orientations based on Fig.B1 of TT2012

    NOTE THAT NOT ALL FEATURES OF FIG.B1 ARE IMPLEMENTED HERE
    """
    i_,_ = idx
    theta_,sigma_,kappa_ = theta[i_],sigma[i_],kappa[i_] 

    # these choices are based on the strike angle
    if abs(theta_ - 90) < EPSVAL:
        return np.where(kappa[idx] < 180)[0]
    elif abs(sigma_ - 90) < EPSVAL:
        return np.where(kappa[idx] < 180)[0]
    elif abs(sigma_ + 90) < EPSVAL:
        return np.where(kappa[idx] < 180)[0]
    else:
        raise Exception


def _fixdet(U):
    if np.linalg.det(U) < 0:
        if VERBOSE > 0:
            print('det(U) < 0: flipping sign of 2nd column')
        U[:,1] *= -1

    return U



### utilities


def _round0(X):
    # round elements near 0
    X[abs(X/max(abs(X))) < EPSVAL] = 0
    return X


def _round1(X):
    # round elements near +/-1
    X[abs(X - 1) < EPSVAL] = -1
    X[abs(X + 1) < EPSVAL] =  1
    return X


def _change_basis(M):
    """ Converts from up-south-east to
        south-east-up convention
    """
    return change_basis(M, i1=1, i2=5)



def _mat(m):
    """ Converts from vector to
        matrix representation
    """
    return np.array(([[m[0], m[3], m[4]],
                      [m[3], m[1], m[5]],
                      [m[4], m[5], m[2]]]))


def _vec(M):
    """ Converts from matrix to
        vector representation
    """
    return np.array([M[0,0], 
                     M[1,1],
                     M[2,2],
                     M[0,1],
                     M[0,2],
                     M[1,2]])


