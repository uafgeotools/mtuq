
import numpy as np

from mtuq.util.moment_tensor import basis
from mtuq.util.lune import fixdet, frame2angles, lam2lune, lune2lam
from mtuq.util.math import eig, rotmat, rotmat_gen


def from_mij(M):
    """
    Converts up-south-east moment tensor to TapeTape2012 parameters

    input: M: moment tensor with shape [6]
              must be in up-south-east (GCMT) convention

    output: gamma, delta, M0, kappa, theta, sigma
    """
    # diagonalize
    lam, U, = eig(_mat(_change_basis(M)), sort_type=1)
    gamma, delta, M0, = lam2lune(lam)

    # require det(U) = 1
    U = fixdet(U)

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



def to_mij(*args):
    """
    Converts TapeTape2012 parameters to up-south-east moment tensor

    input: gamma, delta, M0, kappa, theta, sigma

    output: M: moment tensor with shape [6]
               in up-south-east (GCMT) convention
    """
    try:
        gamma, delta, M0, kappa, theta, sigma = args
    except:
        gamma, delta, M0, kappa, theta, sigma =\
             args[0].gamma, args[0].delta, args[0].M0,\
             args[0].kappa, args[0].theta, args[0].sigma

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
    M = basis.change(_vec(M), 5, 1)

    return M



def _round0(X, thresh=1.e-3):
    # round elements near 0
    X[abs(X/max(abs(X))) < thresh] = 0
    return X


def _round1(X, thresh=1.e-3):
    # round elements near +/-1
    X[abs(X - 1) < thresh] = -1
    X[abs(X + 1) < thresh] =  1
    return X


def _mat(m):
    return np.array(([[m[0], m[3], m[4]],
                      [m[3], m[1], m[5]],
                      [m[4], m[5], m[2]]]))


def _vec(M):
    return np.array([M[0,0],
                     M[1,1],
                     M[2,2],
                     M[0,1],
                     M[0,2],
                     M[1,2]])


def _change_basis(M):
    """ Converts from up-south-east to
        south-east-up convention
    """
    return basis.change(M, i1=1, i2=5)

