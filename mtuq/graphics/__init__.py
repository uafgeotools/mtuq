
import shutil
import subprocess

from mtuq.graphics.beachball import plot_beachball, misfit_vs_depth
from mtuq.graphics.uq import plot_misfit_dc, plot_misfit_vw
from mtuq.graphics.uq_gmt import plot_misfit, plot_likelihood
from mtuq.graphics.waveform import plot_data_synthetics, plot_data_greens


def gmt_version():
    if shutil.which('gmt'):
        proc = subprocess.Popen('gmt --version',
            stdout=subprocess.PIPE, shell=True)

        bytes_string = proc.stdout.readline()
        string = str(bytes_string, "utf-8").strip()
        return string


def gmt_major_version():
    if gmt_version() is not None:
        return int(gmt_version().split('.')[0])


