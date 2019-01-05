
from pkg_resources import iter_entry_points


def readers():
    for entry_point in iter_entry_points('readers'):
        readers[entry_point.name] = entry_point.load()
    return readers

def read(filename_or_url, format='sac', id=None):
    format = format.upper()
    return readers()[format](id=id, **kwargs)

