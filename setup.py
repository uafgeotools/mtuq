from __future__ import print_function
import argparse
import os
import sys
import numpy
from setuptools import find_packages, setup, Extension
from setuptools.command.test import test as test_command


def get_compile_args():
    compiler = ''
    compile_args = []

    try:
        compiler = os.environ["CC"]
    except KeyError:
        pass

    if compiler.endswith("icc"):
        compile_args += ['-fast']
        compile_args += ['-march=native']
    else:
        compile_args += ['-Ofast']
        compile_args += ['-march=native']

    return compile_args


class PyTest(test_command):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


ENTRY_POINTS = {
    'readers': [
        'SAC = mtuq.io.readers.SAC:read',
        ],
    'greens_tensor_clients': [
        'AXISEM = mtuq.io.clients.AxiSEM_NetCDF:Client',
        'AXISEM_NETCDF = mtuq.io.clients.AxiSEM_NetCDF:Client',
        'FK = mtuq.io.clients.FK_SAC:Client',
        'FK_SAC = mtuq.io.clients.FK_SAC:Client',
        'SPECFEM3D = mtuq.io.clients.SPECFEM3D_SAC:Client',
        'SPECFEM3D_SAC = mtuq.io.clients.SPECFEM3D_SAC:Client',
        'SPECFEM3D_SGT = mtuq.io.clients.SPECFEM3D_SGT:Client',
        'SPECFEM3D_PKL = mtuq.io.clients.SPECFEM3D_SGT:Client',
        'SYNGINE = mtuq.io.clients.syngine:Client',
        ]
    }


setup(
    name="mtuq",
    version="0.2.0",
    license='BSD2',
    description="moment tensor (mt) uncertainty quantification (uq)",
    author="Ryan Modrak",
    url="https://github.com/uafgeotools/mtuq",
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology"
    ],
    entry_points=ENTRY_POINTS,
    python_requires='>=3.7.0',
    install_requires=[
        "numpy==1.21",  # workaround for ObsPy regression
        "scipy", "obspy",
        "pandas", "xarray", 
        "retry", 
        "flake8>=3.0", "pytest", "nose", "seisgen",
    ],
    ext_modules = [
        Extension(
            'mtuq.misfit.waveform.c_ext_L2', ['mtuq/misfit/waveform/c_ext_L2.c'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=get_compile_args()),
    ],
)

