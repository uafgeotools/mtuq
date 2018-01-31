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

def iterable(arg):
    if not isinstance(arg, (list, tuple)):
        return [arg]
    else:
        return arg



