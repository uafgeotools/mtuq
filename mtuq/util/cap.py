
import csv


def identifier(stats):
    return '.'.join((
        stats.network,
        stats.station,
        stats.location))


def parse_cap_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights

