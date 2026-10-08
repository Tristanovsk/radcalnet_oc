Installation
============

Look-up tables
--------------

The radiative transfer look-up tables (LUT) of the atmosphere are not part of the package. Download
them from `lutdata <https://drive.google.com/drive/folders/1N0-FtW-PTPblR4z-82fFrUTekMd8e3Vz?usp=sharing>`__
and save them in a directory of your choice (``your_LUT_PATH`` below). If the LUT file is not found, a
light version shipped with the package is used, with a lower accuracy.

The gaseous absorption LUTs, the solar spectra and the irradiance transmittance LUT are included in
the package (``radcalnet_oc/data``).

Python environment
------------------

Python >= 3.9 is required (3.11 or 3.12 recommended). With conda:

.. code-block:: bash

   conda create -n radcalnet_oc python=3.12
   conda activate radcalnet_oc

Configuration
-------------

Set the path of the LUTs in ``radcalnet_oc/config.yml`` before installing:

.. code-block:: yaml

   path:
     lutdata: 'your_LUT_PATH'
     toa_lut: 'toa_lut_opac_wind_v3.nc'
     trans_lut: 'transmittance_lut_opac_wind_v3.nc'

   settings:
     aerosol_combination: [0, 0.5, 0, 0.5, 0., 0.]
     aerosol_models: ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70', 'WASO_rh0']

   processor:
     ncpu: 8
     chunk: 512
     netcdf_engine: 'netcdf4'

``aerosol_combination`` gives the default proportions of the ``aerosol_models`` (see
:ref:`methods-aerosol`).

Install the package
-------------------

From the root of the repository:

.. code-block:: bash

   pip install .

Build this documentation
------------------------

.. code-block:: bash

   pip install ".[docs]"
   sphinx-build -b html docs/source docs/build/html

The tutorial notebooks are not executed: they are rendered with the outputs saved in them.
