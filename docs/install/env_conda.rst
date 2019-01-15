Installation under conda
========================

1. Download and install miniconda following `these instructions <https://conda.io/docs/user-guide/install/index.html>`_.


2. Create a conda virtual environment:

.. code::

   conda create -n mtuq_env python=2
   conda install numpy obspy h5py


3. Install MTUQ: 

.. note::

    If you prefer a location other than ``$HOME/packages``, then modify the following accordinly.


.. code::

   conda activate mtuq_env
   cd $HOME/packages
   git clone https://github.com/uafseismo/mtuq.git
   cd mtuq
   pip install -e .

