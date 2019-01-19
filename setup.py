from __future__ import print_function
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as test_command


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
        'SAC = mtuq.io.readers.sac:read',
        ],
    'greens_tensor_clients': [
        'AXISEM = mtuq.io.greens_tensor.axisem_netcdf:Client',
        'FK = mtuq.io.greens_tensor.fk_sac:Client',
        'SYNGINE = mtuq.io.greens_tensor.syngine:Client',
        ]
    }


setup(
    name="mtuq",
    version="0.2.0",
    license='BSD2',
    description="moment tensor (mt) uncertainty quantification (uq)",
    author="Ryan Modrak",
    author_email="rmodrak@uaf.edu",
    url="https://github.com/uafseismo/mtuq",
    packages=find_packages(),
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
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
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology"
    ],
    entry_points=ENTRY_POINTS,
    python_requires='~=2.7',
    install_requires=[
        "numpy==1.15.4", "scipy", "obspy", "h5py", "retry",
        "flake8>=3.0", "pytest", "nose"
    ]
)
