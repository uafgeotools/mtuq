
import numpy as np

PI = np.pi
DEG = 180./PI
INF = np.inf


def eig(M, sort_type=1):
    """
    Calculates eigenvalues and eigenvectors of matrix
    """
    if sort_type not in [1,2,3,4]:
        raise ValueError

    lam,V = np.linalg.eigh(M)

    # sorting of eigenvalues
    # 1: highest to lowest, algebraic: lam1 >= lam2 >= lam3
    # 2: lowest to highest, algebraic: lam1 <= lam2 <= lam3
    # 3: highest to lowest, absolute : | lam1 | >= | lam2 | >= | lam3 |
    # 4: lowest to highest, absolute : | lam1 | <= | lam2 | <= | lam3 |
    if sort_type == 1:
        idx = np.argsort(lam)[::-1]
    elif sort_type == 2:
        idx = np.argsort(lam)
    elif sort_type == 3:
        idx = np.argsort(np.abs(lam))[::-1]
    elif sort_type == 4:
        idx = np.argsort(np.abs(lam))
    lsort = lam[idx]
    Vsort = V[:,idx]

    return lsort,Vsort



def rotmat(xdeg, idx):
    """ 3D rotation matrix about given axis
    """
    if idx not in [0, 1, 2]:
        raise ValueError

    cosx = np.cos(xdeg / DEG)
    sinx = np.sin(xdeg / DEG)

    if idx==0:
        return np.array([
            [1, 0, 0],
            [0, cosx, -sinx],
            [0, sinx, cosx],
            ])

    elif idx==1:
        return np.array([
            [cosx, 0, sinx],
            [0, 1, 0],
            [-sinx, 0, cosx],
            ])

    elif idx==2:
        return np.array([
            [cosx, -sinx, 0],
            [sinx, cosx, 0],
            [0, 0, 1],
            ])


def rotmat_gen(v, xi):
    rho = np.linalg.norm(v)
    vth = np.arccos(v[2] / rho)
    vph = np.arctan2(v[1],v[0])

    return np.dot(np.dot(np.dot(np.dot(
            rotmat(vph*DEG,2),
            rotmat(vth*DEG,1)),
            rotmat(xi,2)),
            rotmat(-vth*DEG,1)),
            rotmat(-vph*DEG,2))


def fangle(x,y):
    """ Returns the angle between two vectors, in degrees
    """
    xy = np.dot(x,y)
    xx = np.dot(x,x)
    yy = np.dot(y,y)
    return np.arccos(xy/(xx*yy)**0.5)*DEG



def fangle_signed(va,vb,vnor):
    """ Returns the signed angle (of rotation) between two vectors, in degrees
    """

    # get rotation angle (always positive)
    theta = fangle(va,vb);

    EPSVAL = 0;
    stheta = theta;     # initialize to rotation angle
    if abs(theta - 180) <= EPSVAL:
        stheta = 180
    else:
        Dmat = np.column_stack([va, vb, vnor])
        if np.linalg.det(Dmat) < 0:
            stheta = -theta

    return stheta



def wrap360(omega):
    """ Wrap phase
    """
    return omega % 360.



def isclose(X, Y):
    EPSVAL = 1.e-6
    X = np.array(X)
    Y = np.array(Y)
    return bool(
        np.linalg.norm(X-Y) < EPSVAL)



def open_interval(x1,x2,nx):
    return np.linspace(x1,x2,n+2)[1:-1]



def closed_interval(x1,x2,nx):
    return np.linspace(x1,x2,n)

