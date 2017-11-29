
def factory(type, *args, **kwargs):
    module = __import__('mtuq.greens1d.'+type)
    cls = getattr(module, 'generator')
    return cls(*args, **kwargs)

