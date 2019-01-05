
from pkg_resources import iter_entry_points


def _greens_databases():
    databases = {}
    for entry_point in iter_entry_points('greens_databases'):
        databases[entry_point.name] = entry_point.load()
    return databases

def open_greens_db(format='', **kwargs):
    format = format.upper()
    return _greens_databases()[format](**kwargs)

