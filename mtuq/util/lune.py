
import numpy as np


kR3 = np.sqrt(3.0)
k2R6 = 2.0 * np.sqrt(6.)
k2R3 = 2.0 * np.sqrt(3.)
k4R6 = 4.0 * np.sqrt(6.)
k8R6 = 8.0 * np.sqrt(6.)
deg2rad = np.pi / 180.
beta0 = np.linspace(0, np.pi, 100)
u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)


def to_mij(rho, v, w, kappa, sigma, h):
    """ Converts from lune parameters to moment tensor parameters 
    (up-south-east convention)
    """
    m0 = rho/np.sqrt(2.)

    gamma = (1./3.)*np.arcsin(3.*v)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    gamma /= deg2rad
    beta /= deg2rad

    delta = 90. - beta
    theta = np.arccos(h)/deg2rad

    Cb  = np.cos(beta*deg2rad)
    Cg  = np.cos(gamma*deg2rad)
    Cs  = np.cos(sigma*deg2rad)
    Ct  = np.cos(theta*deg2rad)
    Ck  = np.cos(kappa*deg2rad)
    C2k = np.cos(2.0*kappa*deg2rad)
    C2s = np.cos(2.0*sigma*deg2rad)
    C2t = np.cos(2.0*theta*deg2rad)

    Sb  = np.sin(beta*deg2rad)
    Sg  = np.sin(gamma*deg2rad)
    Ss  = np.sin(sigma*deg2rad)
    St  = np.sin(theta*deg2rad)
    Sk  = np.sin(kappa*deg2rad)
    S2k = np.sin(2.0*kappa*deg2rad)
    S2s = np.sin(2.0*sigma*deg2rad)
    S2t = np.sin(2.0*theta*deg2rad)

    mt0 = m0 * (1./12.) * \
        (k4R6*Cb + Sb*(kR3*Sg*(-1. - 3.*C2t + 6.*C2s*St*St) + 12.*Cg*S2t*Ss))

    mt1 = m0* (1./24.) * \
        (k8R6*Cb + Sb*(-24.*Cg*(Cs*St*S2k + S2t*Sk*Sk*Ss) + kR3*Sg * \
        ((1. + 3.*C2k)*(1. - 3.*C2s) + 12.*C2t*Cs*Cs*Sk*Sk - 12.*Ct*S2k*S2s)))

    mt2 = m0* (1./6.) * \
        (k2R6*Cb + Sb*(kR3*Ct*Ct*Ck*Ck*(1. + 3.*C2s)*Sg - k2R3*Ck*Ck*Sg*St*St + 
        kR3*(1. - 3.*C2s)*Sg*Sk*Sk + 6.*Cg*Cs*St*S2k + 
        3.*Ct*(-4.*Cg*Ck*Ck*St*Ss + kR3*Sg*S2k*S2s)))

    mt3 = m0* (-1./2.)*Sb*(k2R3*Cs*Sg*St*(Ct*Cs*Sk - Ck*Ss) +
        2.*Cg*(Ct*Ck*Cs + C2t*Sk*Ss))

    mt4 = -m0* (1./2.)*Sb*(Ck*(kR3*Cs*Cs*Sg*S2t + 2.*Cg*C2t*Ss) +
        Sk*(-2.*Cg*Ct*Cs + kR3*Sg*St*S2s))

    mt5 = -m0* (1./8.)*Sb*(4.*Cg*(2.*C2k*Cs*St + S2t*S2k*Ss) +
        kR3*Sg*((1. - 2.*C2t*Cs*Cs - 3.*C2s)*S2k + 4.*Ct*C2k*S2s))

    if type(mt0) is np.ndarray:
        return np.column_stack([mt0, mt1, mt2, mt3, mt4, mt5])
    else:
        return np.array([mt0, mt1, mt2, mt3, mt4, mt5])


def spherical_to_Cartesian(r, theta, phi):
    """ Converts from spherical to Cartesian coordinates
    """
    x = r*np.sin(theta)*cos(phi)
    y = r*np.sin(theta)*cos(phi)
    z = r*np.cos(theta)

    return np.array([x, y, z])


def to_xyz(F0, theta, h):
    """ Converts from spherical to Cartesian coordinates
    """
    r = F0
    phi = np.arcos(h)

    return spherical_to_Cartesian(r, theta, phi)




