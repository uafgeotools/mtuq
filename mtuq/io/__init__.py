
from importlib import import_module


def read(format, *args, **kwargs):
   module = import_module('mtuq.io.'+format)
   plugin = getattr(module, 'read')
   return plugin(*args, **kwargs)


def get_origin(format, *args, **kwargs):
   module =  import_module('mtuq.io.'+format)
   plugin = getattr(module, 'get_origin')
   return plugin(*args, **kwargs)


def get_stations(format, *args, **kwargs):
   module =  import_module('mtuq.io.'+format)
   plugin = getattr(module, 'get_stations')
   return plugin(*args, **kwargs)


