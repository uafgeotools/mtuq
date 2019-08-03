
import numpy as np

from mtuq.util.moment_tensor import TapeTape2012
from mtuq.util.lune import lune2rect, rect2lune
from mtuq.util.math import DEG, rotmat, rotmat_gen



def from_mij(M):
    """
    Converts up-south-east moment tensor to 2015 parameters

    input: M: moment tensor with shape [6]
              must be in up-south-east (GCMT) convention

    output: kappa, sigma, M0, v, w, h
    """
    gamma, delta, M0, kappa, theta, sigma = TapeTape2012.from_mij(M)
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


def to_mij(*args):
    """
    Converts 2015 parameters to up-south-east moment tensor

    input: kappa, sigma, M0, v, w, h

    output: M: moment tensor with shape [6]
               in up-south-east (GCMT) convention
    """
    try:
        rho, v, w, kappa, sigma, h = args
    except:
        rho, v, w, kappa, sigma, h =\
            args[0].rho, args[0].v, args[0].w,\
            args[0].kappa, args[0].sigma, args[0].h

    theta = np.arccos(h)*DEG
    M0 = rho/np.sqrt(2)
    gamma, delta = rect2lune(v, w)
    M = TapeTape2012.to_mij(gamma, delta, M0, kappa, theta, sigma)
    return M
