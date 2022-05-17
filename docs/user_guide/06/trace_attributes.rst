
Trace attributes
================

During a seismic inversion waveform differences, phase shifts, amplitude ratios, and other quantities are calculated from observed and synthetic seismic traces. 

Such `trace attribute` quantities provide important information about how data misfit varies by geographic location and seismic component or with repsect to particular inversion settings. 


Collecting trace attributes
---------------------------

Through the misfit function's `collect_attributes <https://uafgeotools.github.io/mtuq/library/generated/mtuq.Misfit.collect_attributes.html#mtuq.Misfit.collect_attributes>`_ method, MTUQ provides a way to collect time shift, amplitude ratio and other trace attributes.

Suppose we have the following misfit function:

.. code::

    misfit = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        )


To get a data structure containing trace attributes, we can invoke the misfit function's `collect_attributes` method using the same input arguments as we would for an ordinary misfit function evaluation:

.. code::

    attrs =  misfit.collect_attributes(data, greens, source)


The `attrs` data structure above consists of a list of `AttribDict <https://docs.obspy.org/packages/autogen/obspy.core.util.attribdict.AttribDict.html>`_ objects organized by station and component.


Attaching trace attributes to `Datasets`
----------------------------------------

We can also attach trace attributes to the observed data directly by invoking the misfit function as follows:

.. code::

    misfit(data, greens, source, set_attributes=True)


During evaluation of the above, a `AttribDict` will be attached to each `Trace` object in the `Dataset`.

.. note::

    With `set_attribtues=True`, it only makes sense to pass a single `source` to the misfit function (because if multiple sources are passed, attributes from all but the last source end up getting overwritten).


Custom trace attributes
-----------------------

Users can add additional trace attributes by modifying `mtuq/misfit/waveform/level0.py <https://github.com/uafgeotools/mtuq/blob/master/mtuq/misfit/waveform/level0.py>`_. Any changes made here carry over to both the `collect_attributes` and `set_attributes` methods above.


