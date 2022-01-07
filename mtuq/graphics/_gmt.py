
import os
import numpy as np
import re
import shutil
import subprocess

from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb, Normalize
from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180
from os.path import splitext


gmt_colors = {
    # Name:  (R   G   B)
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'yellow': (255, 255, 0),
    'red': (255, 0, 0),
    }


gmt_formats = [
    'BMP',
    'EPS',
    'JPG',
    'JPEG',
    'PDF',
    'PNG',
    'PPM',
    'SVG',
    'TIFF',
    ]


def exists_gmt():
    return bool(shutil.which('gmt'))


def gmt_version():
    if exists_gmt():
        proc = subprocess.Popen('gmt --version',
            stdout=subprocess.PIPE, shell=True)

        bytes_string = proc.stdout.readline()
        string = str(bytes_string, "utf-8").strip()
        return string


def gmt_major_version():
    if gmt_version() is not None:
        return int(gmt_version().split('.')[0])


def gmt_not_found_warning(filename):
    warn("""
        WARNING

        Generic Mapping Tools executables not found on system path.
        PostScript output has not been written. 

        Misfit values have been saved to:
            %s
        """ % filename)


def read_cpt(filename, name=None, N=256):
    if name is None:
        name = os.path.basename(filename).split('.')[0]

    with open(filename, 'r') as file:
        color_model = "RGB"

        cpt = []
        for line in file:
            line = line.strip()

            if not line:
                continue

            if "HSV" in line:
                color_model = "HSV"
                continue

            if line.startswith('#'):
                continue

            if line.startswith(("B", "F", "N")):
                continue

            cpt.append(line)

    return _parse_cpt(cpt, name, color_model, N)


def _parse_cpt(cpt, name, color_model='RGB', N=256):
    x = []
    r = []
    g = []
    b = []
    for segment in cpt:
        fields = re.split(r'\s+|[/]', segment)
        x.append(float(fields[0]))
        try:
            r.append(float(fields[1]))
            g.append(float(fields[2]))
            b.append(float(fields[3]))
            xi = 4
        except ValueError:
            r_, g_, b_ = gmt_colors[fields[1]]
            r.append(float(r_))
            g.append(float(g_))
            b.append(float(b_))
            xi = 2

    x.append(float(fields[xi]))

    try:
        r.append(float(fields[xi + 1]))
        g.append(float(fields[xi + 2]))
        b.append(float(fields[xi + 3]))
    except ValueError:
        r_, g_, b_ = gmt_colors[fields[-1]]
        r.append(float(r_))
        g.append(float(g_))
        b.append(float(b_))

    x = np.array(x)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    if color_model == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = hsv_to_rgb(r[i] / 360., g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    elif color_model == "RGB":
        r /= 255.
        g /= 255.
        b /= 255.
    else:
        raise ValueError('Bad argument: color_model')

    norm = Normalize(vmin=x[0], vmax=x[-1])(x)

    red = []
    blue = []
    green = []
    for i in range(norm.size):
        red.append([norm[i], r[i], r[i]])
        green.append([norm[i], g[i], g[i]])
        blue.append([norm[i], b[i], b[i]])

    cdict = dict(red=red, green=green, blue=blue)
    cmap = LinearSegmentedColormap(name=name, segmentdata=cdict, N=N)
    cmap.values = x
    cmap.colors = list(zip(r, g, b))
    cmap._init()
    return cmap


def _parse_filetype(filename):

    parts = splitext(filename)
    name, ext = parts[0], parts[1].lstrip('.')

    if ext.upper() in ['PS']:
        return name, 'EPS'

    elif ext.upper() in ['JPG', 'JPEG']:
        return name, 'JPEG'

    elif ext.upper() in ['TIF', 'TIFF']:
        return name, 'TIFF'

    elif ext.upper() in gmt_formats:
        return name, ext.upper()

    else:
        warn('Unrecognized extension: defaulting to PNG')
        return name, 'PNG'


def _get_format_arg(fmt):

    assert fmt in gmt_formats

    if fmt=='BMT':
        return '-Tb'

    elif fmt=='EPS':
        return '-Te'

    elif fmt=='PDF':
        return '-Tf'

    elif fmt=='JPEG':
        return '-Tj'

    elif fmt=='PNG':
        return '-Tg'

    elif fmt=='PPM':
        return '-Tm'

    elif fmt=='SVG':
        return '-Ts'

    elif fmt=='TIFF':
        return '-Tt'

