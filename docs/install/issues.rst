

Installation on Apple M1 and Apple M2 Macs
------------------------------------------

A modified installation procedure may be necessary on Apple M1 and Apple M2 Macs.

For more information, please see:

`MTUQ installation on Apple M1 and Apple M2 Macs <https://uafgeotools.github.io/mtuq/install/arm64.html>`_


Installation under virutalenv
-----------------------------

For most users, we recommend installing MTUQ under conda.  In rare cases, however, it seems that MTUQ can be more easily installed under virtualenv.

For more information, please see:

`MTUQ installation under virtualenv <https://uafgeotools.github.io/mtuq/install/virtualenv.html>`_


Troubleshooting PyGMT installation
----------------------------------

To avoid a known issue with PyGMT installation under conda, it is necessary to specify a minimum PyGMT version as follows:

.. code::

    conda install 'pygmt>=0.9'


A more conservative approach, which avoids suspected problems with PyGMT 0.10 releases, is to specify the more stable PyGMT 0.9 version:

.. code::

    conda install 'pygmt=0.9'


If PyGMT installation fails, most MTUQ functions will still work, but some plotting functions will fall back to GMT or ObsPy.  We note that some versions of GMT and ObsPy do not plot `full moment tensors <https://github.com/obspy/obspy/issues/2388>`_ correctly.

