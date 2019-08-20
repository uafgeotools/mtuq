
import numpy as np

PI = np.pi
DEG = 180./PI
INF = np.inf


def list_intersect(a, b):
    """ Intersection of two lists
    """
    return list(set(a).intersection(set(b)))


def list_intersect_with_indices(a, b):
    intersection = list(set(a).intersection(set(b)))
    indices = [a.index(item) for item in intersection]
    return intersection, indices


def isclose(X, Y):
    EPSVAL = 1.e-6
    X = np.array(X)
    Y = np.array(Y)
    return bool(
        np.linalg.norm(X-Y) < EPSVAL)


def open_interval(x1,x2,nx):
    return np.linspace(x1,x2,nx+2)[1:-1]


def closed_interval(x1,x2,nx):
    return np.linspace(x1,x2,nx)

