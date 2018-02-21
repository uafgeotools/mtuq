#!/usr/bin/env python

class AttribDict(dict):
    """ Dictionary with both keyword and attribute access
    """
    def __init__(self, *args, **kwargs):
        super(AttribDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Struct(dict):
    """ Dictionary with both keyword and attribute access
    """
    def __init__(self, *args, **kwargs):
        super(Struct, self).__init__(*args, **kwargs)
        self.__dict__ = self


def is_mpi_env():
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    return True


def iterable(arg):
    if not isinstance(arg, (list, tuple)):
        return [arg]
    else:
        return arg



