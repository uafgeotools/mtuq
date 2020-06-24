
import shutil
import subprocess
from os.path import splitext



def gmt_cmd():
    return shutil.which('gmt')


def gmt_version():
    if gmt_cmd() is not None:
        proc = subprocess.Popen('gmt --version',
            stdout=subprocess.PIPE, shell=True)

        bytes_string = proc.stdout.readline()
        string = str(bytes_string, "utf-8").strip()
        return string


def gmt_major_version():
    if gmt_version() is not None:
        return int(gmt_version().split('.')[0])


def gmt_not_found_warning(filename):
    warnings.warn("""
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


def check_ext(filename):
    name, ext = _splitext(filename)

    if ext.upper() in ['PS']:
        return name, 'EPS'

    elif ext.upper() in ['JPG', 'JPEG']:
        return name, 'JPEG'

    elif ext.upper() in ['TIF', 'TIFF']:
        return name, 'TIFF'

    elif ext.upper() in gmt_formats:
        return name, ext.upper()

    else:
        print('Unrecognized extension: defaulting to PNG')
        return filename, 'PNG'


def _splitext(filename):
   parts = splitext(filename)
   return parts[0], parts[1].lstrip('.')


