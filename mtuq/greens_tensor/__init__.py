
from pkg_resources import iter_entry_points


def greens_dbs():
    for entry_point in iter_entry_points('greens_dbs'):
        greens_dbs[entry_point.name] = entry_point.load()
    return greens_dbs

def open_greens_db(filename_or_url, format=''):
    format = format.upper()
    return greens_dbs()[format](id=id, **kwargs)

