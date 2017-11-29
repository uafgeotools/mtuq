

def get_origin(data, format=None, **kwargs):
   module =  __import__('mtuq.io.'+format)
   plugin = getattr(module, 'get_origin')
   return plugin(data)


def get_stations(data, format=None, **kwargs):
   module =  __import__('mtuq.io.'+format)
   plugin = getattr(module, 'get_stations')
   return plugin(data)


def read(data, format=None, **kwargs):
   module = __import__('mtuq.io.'+format)
   plugin = getattr(module, 'get_origin')
   return plugin(data)

