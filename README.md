# mtuq

MTUQ provides *m*oment *t*ensor estimates and *u*ncertainty *q*uantification from broadband seismic data, along with

- ability to interface with modern solvers, including AxiSEM and SPECFEM3D
- GMT wrappers for uncertainty visualization on the eigenvalue lune
- C-accelerated, MPI-parallelized misfit function evaluation
- a flexible Python design borrowing from [ObsPy] and [Instaseis] data structures

The package has been [tested against](https://github.com/uafgeotools/mtuq/blob/master/tests/benchmark_cap_vs_mtuq.py) community Perl/C codes.




## Getting started

[Installation](https://uafgeotools.github.io/mtuq/install/index.html)

[Quick start](https://uafgeotools.github.io/mtuq/quick_start.html)



## User guide

[Learning Python and ObsPy](https://uafgeotools.github.io/mtuq/user_guide/01.html)

[Acquiring seismic data](https://uafgeotools.github.io/mtuq/user_guide/02.html)

[Acquiring Green's functions](https://uafgeotools.github.io/mtuq/user_guide/03.html)

[Library reference](https://uafgeotools.github.io/mtuq/library/index.html)



[![Build Status](https://travis-ci.org/uafseismo/mtuq.svg?branch=master)](https://travis-ci.org/uafseismo/mtuq)

[Instaseis]: http://instaseis.net/

[obspy]: https://github.com/obspy/obspy/wiki

[ZhaoHelmberger1994]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/84/1/91/102552/Source-estimation-from-broadband-regional?redirectedFrom=fulltext

[ZhuHelmberger1996]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/86/5/1634/120218/Advancement-in-source-estimation-techniques-using?redirectedFrom=fulltext

