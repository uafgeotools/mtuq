
Trace attributes
================

In a seismic inversion, waveform differences, phase shifts, amplitude ratios, and other quantities are calculated from observed and synthetic seismic traces. Such quantities provide important information about how data fit evolves during an inversion and how it varies according to location and component. 


Collecting trace attributes
---------------------------

Through the misfit function's `collect_attributes` method, MTUQ provides a way to collect time shift, amplitude ratio and other trace attributes.

Suppose we have defined a misfit function as follows:

.. code::

    misfit = Misfit(norm='L2')


To get a data structure containing trace attributes, we can invoke `collect_attributes()` using the same input arguments that would for an ordinary misfit function evaluation:

.. code::

    attrs =  misfit.collect_attributes(data, greens, source)


The `attrs` data structure in the above code snippet consists of a list of `AttribDict` objects organized by station and component. For an example of how it can be used in practice, see `examples/DetailedAnalysis.FullMomentTensor.py`.


Attaching trace attributes to `Datasets`
----------------------------------------

We can also attach traces attributes to the observed data directly by invoking the misfit function as follows:

.. code::

    misfit(data, greens, source, set_attributes=True)


During misfit evaluation, a `AttribDict` will be attached to each `Trace` object in the `Dataset`.

.. note::

    With `set_attribtues=True`, it only makes sense to pass a single source to the misfit function. If multiple sources are passed, then attributes from all but the last source end up getting overwritten.


Custom trace attributes
-----------------------

Users can add additional trace attributes by modifying `mtuq/misfit/waveform/level0.py`. Any changes made to the `AttribDict` objects here carry over to both the `collect_attributes` and `set_attributes` methods above.


