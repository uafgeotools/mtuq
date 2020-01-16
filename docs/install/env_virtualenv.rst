Installation under virtualenv
=============================

Install the `virtualenv <https://virtualenv.pypa.io/en/latest/installation/>`_ program, if not already present.

Create a new virtual environment:

.. code::

   virtualenv --python=python3 mtuq_env


Download MTUQ source code:

.. code::

   git clone https://github.com/uafgeotools/mtuq.git mtuq_src


Install MTUQ:

.. code::

   source mtuq_env/bin/activate
   cd mtuq_src
   pip install -e .



