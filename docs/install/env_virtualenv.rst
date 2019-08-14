Installation under virtualenv
=============================

Install the `virtualenv <https://virtualenv.pypa.io/en/latest/installation/>`_ program, if not already present.

Create a new virtual environment:

.. note::

    If you prefer a location other than ``$HOME``, then change the first line accordingly.

.. code::

   cd $HOME
   virtualenv --python=python2.7 mtuq_env


Download MTUQ source code:

.. code::

   git clone https://github.com/uafseismo/mtuq.git


Install MTUQ:

.. code::

   source mtuq_env/bin/activate
   cd mtuq
   pip install -e .



