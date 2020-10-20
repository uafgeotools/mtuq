
import numpy as np
import shutil
import subprocess

from mtuq.util import fullpath, warn
from mtuq.util.math import wrap_180
from os.path import splitext



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


