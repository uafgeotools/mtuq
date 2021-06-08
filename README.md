# mtuq

MTUQ provides *m*oment *t*ensor estimates and *u*ncertainty *q*uantification from broadband seismic data.  

Use cases include full moment tensor anaysis, constrained moment tensor analysis, and many other possibilities involving moment tensor, force, hypocenter, depth, and wavelet parameters.

To provide these capbilities, MTUQ draws on ObsPy, instaseis, and pandas data structures.


## Misfit evaluation

Waveform difference and cross-correlation time-shift [misfit evaluation](https://uafgeotools.github.io/mtuq/library/index.html#data-processing-and-inversion) 
on body-wave and surface-wave windows is implemented in C-accelerated Python.

These misfit functions can be used with [mtuq.grid_search](https://uafgeotools.github.io/mtuq/library/generated/mtuq.grid_search.grid_search.html), which automatically partitions the grid over multiple MPI processes if invoked from an MPI environment.  For efficient and unbiased uncertainty quantification, [uniform grids](https://uafgeotools.github.io/mtuq/library/index.html#moment-tensor-and-force-grids) can be used for the grid search, drawing from [Tape2015](https://academic.oup.com/gji/article/202/3/2074/613765).

Alternatively, MTUQ misfit functions can be used as a starting point for Bayesian uncertainty quantification using pymc or other MCMC libraries.


## Solver interfaces

[Data I/O functions](https://uafgeotools.github.io/mtuq/library/index.html#data-i-o)
are included for reading AxiSEM, SPECFEM3D, and FK Green's functions as well as
downloading Green's functions from remote syngine databases.


## Visualization

[Visualization utilities](https://uafgeotools.github.io/mtuq/library/index.html#uncertainty-visualization) are included for both the [eigenvalue lune](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-246X.2012.05491.x) and [v-w rectangle](https://academic.oup.com/gji/article/202/3/2074/613765), with matplotlib and Generic Mapping Tools graphics backends.


## Testing

The package has been tested against [legacy Perl/C codes](https://github.com/uafgeotools/mtuq/blob/master/tests/benchmark_cap_vs_mtuq.py) as well as [published studies](https://github.com/rmodrak/mtbench).




## Getting started

[Installation](https://uafgeotools.github.io/mtuq/install/index.html)

[Quick start](https://uafgeotools.github.io/mtuq/quick_start.html)



## Documentation

[Acquiring seismic data](https://uafgeotools.github.io/mtuq/user_guide/02.html)

[Acquiring Green's functions](https://uafgeotools.github.io/mtuq/user_guide/03.html)

[Data processing](https://uafgeotools.github.io/mtuq/user_guide/04.html)

[Visualization galleries](https://uafgeotools.github.io/mtuq/user_guide/05.html)

[Library reference](https://uafgeotools.github.io/mtuq/library/index.html)



[![Build Status](https://travis-ci.org/uafgeotools/mtuq.svg?branch=master)](https://travis-ci.org/uafgeotools/mtuq)

[Instaseis]: http://instaseis.net/

[obspy]: https://github.com/obspy/obspy/wiki

[ZhaoHelmberger1994]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/84/1/91/102552/Source-estimation-from-broadband-regional?redirectedFrom=fulltext

[ZhuHelmberger1996]: https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/86/5/1634/120218/Advancement-in-source-estimation-techniques-using?redirectedFrom=fulltext

