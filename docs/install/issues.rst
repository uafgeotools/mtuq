
Troubleshooting
===============


PyGMT installation
------------------

MTUQ uses full moment tensor plotting functions from the PyGMT graphics library.

To avoid a known issue with PyGMT installation under conda, it is necessary to specify a minimum PyGMT version as follows:

.. code::

    conda install 'pygmt>=0.9'


A more conservative approach, which avoids possible issues with PyGMT 0.10 releases, is to specify the more stable PyGMT 0.9 version:

.. code::

    conda install 'pygmt=0.9'


If PyGMT installation fails, most MTUQ functions will still work, but some plotting functions will fall back to GMT or ObsPy.  

We note that some versions of GMT and ObsPy do not plot `full moment tensors <https://github.com/obspy/obspy/issues/2388>`_ correctly.



Speeding up conda installs
--------------------------

Sometimes the conda package manager can be very slow. To speed it up, older conda installations can be updated, making libmamba the default dependency solver:

.. code::

    conda update -n base conda



MTUQ installation on Apple M1 and Apple M2 Macs
-----------------------------------------------

Installation on Apple M1 and Apple M2 Macs is now possible using the default installation procedure.

For older versions of MTUQ, a modified installation procedure may stil be necessary.  For more information, please see:

`MTUQ installation on ARM64 systems <https://uafgeotools.github.io/mtuq/install/arm64.html>`_

