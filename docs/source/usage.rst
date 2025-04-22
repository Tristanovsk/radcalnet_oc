Usage
=====

Installation
------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.

Download the LUT files
~~~~~~~~~~~~~~~~~~~~~~~

click on
`lutdata <https://drive.google.com/drive/folders/1N0-FtW-PTPblR4z-82fFrUTekMd8e3Vz?usp=sharing>`__
to download and save in your desired path (your_LUT_PATH)

Installation with conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   conda activate "name of your conda env"

Python >= 3.9 is recommended, example:

::

   conda create python=3.12 -n your_env
   conda activate your_env


Set the ``config.yml`` file:

::

   path:
     lutdata: your_LUT_PATH

Finally, install grs with:

.. code::

   pip install .

Testing 
-------

TODO