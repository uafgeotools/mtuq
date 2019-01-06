
from pkg_resources import iter_entry_points


def _readers():
    readers = {}
    for entry_point in iter_entry_points('readers'):
        readers[entry_point.name] = entry_point.load()
    return readers

def read(filename_or_url, format='sac', id=None, **kwargs):
    format = format.upper()
    return _readers()[format](filename_or_url, id=id, **kwargs)

