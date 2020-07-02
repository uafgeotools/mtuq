# mtuq

MTUQ provides *m*oment *t*ensor estimates and *u*ncertainty *q*uantification from broadband seismic data, drawing on ObsPy and instaseis data structures.


## Solvers

Interfaces are provided for modern solvers, including AxiSEM and SPECFEM3D, 
and legacy solves, including FK.

[Utilities](https://uafgeotools.github.io/mtuq/library/index.html#data-i-o)
are included for working with AxiSEM, SPECFEM3D, and FK databases as well as
downloading AxiSEM synthetics from remote syngine databases.


## Misfit evaluation

Waveform difference and cross-correlation time-shift [misfit evaluation](https://uafgeotools.github.io/mtuq/library/index.html#data-processing-and-inversion) 
on windowed body-wave and surface-wave traces is implemented in C-accelerated 
Python.

A separate easy-to-read pure Python implementation is also included for 
checking the correctness of the accelerated version.


## Visualization

Includes moment tensor [uncertainty quantification](https://uafgeotools.github.io/mtuq/library/index.html#uncertainty-quantification) on the eigenvalue lune and
v-w rectangle, with separate matplotlib and Generic Mapping Tools plotting
functions.


## Testing

The package has been tested against [legacy Perl/C codes](https://github.com/uafgeotools/mtuq/blob/master/tests/benchmark_cap_vs_mtuq.py) and [published inversions](https://github.com/rmodrak/mtbench).




## Getting started

[Installation](https://uafgeotools.github.io/mtuq/install/index.html)

[Quick start](https://uafgeotools.github.io/mtuq/quick_start.html)



## User guide

[Learning Python and ObsPy](https://uafgeotools.github.io/mtuq/user_guide/01.html)

[Acquiring seismic data](https://uafgeotools.github.io/mtuq/user_guide/02.html)

[Acquiring Green's functions](https://uafgeotools.github.io/mtuq/user_guide/03.html)

[Library reference](https://uafgeotools.github.io/mtuq/library/index.html)



[![Build Status](https://travis-ci.org/uafgeotools/mtuq.svg?branch=master)](https://travis-ci.org/uafseismo/mtuq)

[Instaseis]: http://instaseis.net/

[obspy]: https://github.com/obspy/obspy/wiki

[ZhaoHelmberger1994]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/84/1/91/102552/Source-estimation-from-broadband-regional?redirectedFrom=fulltext

[ZhuHelmberger1996]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/86/5/1634/120218/Advancement-in-source-estimation-techniques-using?redirectedFrom=fulltext

