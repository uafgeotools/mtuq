
Plotting time shifts and amplitude ratios
=========================================

Following `Zhao1994 <https://uafgeotools.github.io/mtuq/references.html>`_, many moment tensor inversions use time shifts to mitigate velocity model inaccuracy. 

It is common to plot how these time shifts vary with geographic location.


Keeping track of trace attributes
---------------------------------

To keep track of time shifts and other trace attributes, MTUQ uses `this approach <https://uafgeotools.github.io/mtuq/user_guide/06/trace_attributes.html>`_, in which the `collect_attributes` method returns a data structure containing time shifts and other information:

.. code::

    attrs = misfit.collect_attributes(data, greens, source)


Plotting time shifts
--------------------

To plot time shifts, the `attrs` data structure returned above can simply be passed to a plotting function:

.. code::

    from mtuq.graphics import plot_time_shifts
    plot_time_shifts(dirname, attrs, stations, origins, components=['Z','R','T'])

As a result, one image like the following will be created for each component.

.. image:: images/20090407201255351_attrs_time_shifts_bw_Z.png
  :width: 400 


Plotting amplitude ratios
-------------------------

Similarly, the `attrs` data structure can be used to plot amplitude ratios:

.. code::

    from mtuq.graphics import plot_amplitude_ratios
    plot_amplitude_ratios(dirname, attrs, stations, origins, components=['Z','R','T'])

As a result, one image like the following will be created for each component

.. image:: images/20090407201255351_attrs_amplitude_ratios_bw_Z.png
  :width: 400 


Plotting custom attributes
--------------------------

It is possible to `add custom trace attributes <https://uafgeotools.github.io/mtuq/user_guide/06/trace_attributes.html#custom-trace-attributes>`_ to the data structure returned by `collect_attributes`. 

To plot how a custom attribute (say, `custom_attr`) varies, we can use the private function `_plot_attrs`:

.. code::

    from mtuq.graphics import _plot_attrs
    _plot_attrs(dirname, stations, origins, attrs, 'custom_attr', components=['Z','R','T'])


It turns out that `_plot_attr` is also what `plot_time_shifts`, `plot_amplitude_ratios`, `plot_log_amplitude_ratios` use under the hood to create figures.


Source code
-----------

The above figures can be reproduced by running `DetailedAnalysis.FullMomentTensor.py <https://github.com/uafgeotools/mtuq/blob/master/examples/DetailedAnalysis.FullMomentTensor.py>`_.

