
Customizing figures
===================

MTUQ plotting functions aim for generality and ease of use. Sometimes, this results in a default figure that is a bit generic. 

Fortunately, figure appearance is highly customizable, as illustrated below.


Time shift and amplitude ratio figures
---------------------------------------

To plot how time shifts, amplitude ratios, or other `trace attributes <https://uafgeotools.github.io/mtuq/user_guide/06/trace_attributes.html>`_ vary geographically, we can pass the output of `collect_attributes <https://uafgeotools.github.io/mtuq/library/generated/mtuq.Misfit.collect_attributes.html#mtuq.Misfit.collect_attributes>`_ to a plotting function, as in the following example:

.. code::

    plot_time_shifts(dirname, attrs, stations, origins)


Because we haven't specified a map projection or other details, it's no surprise that the resulting figure is quite generic:

.. image:: images/20090407201255351_attrs_time_shifts_bw_Z.png
  :width: 400 


To some extent, the above figure can be customized through optional arguments, which get passed to a backend function `_plot_attrs` (see `documentation <https://uafgeotools.github.io/mtuq/library/generated/mtuq.graphics.plot_time_shifts.html>`_).

A much more powerful approach, though, is to overload the backend entirely:

.. code::

    plot_time_shifts(dirname, attrs, stations, origins, backend=user_supplied_function)

To illustrate, we have implemented a `custom backend <>`_ for the `Silwal2016` study area. The result of calling `plot_time_shifts` with this new backend is shown below:


.. image:: images/20090407201255351_attrs_time_shifts_bw_Z_pygmt.png
  :width: 400 

