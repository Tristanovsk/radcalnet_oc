Installation
============

Python environment
------------------

Python >= 3.10 is required (3.10, 3.11 and 3.12 are tested). With conda:

.. code-block:: bash

   conda create -n radcalnet_oc python=3.12
   conda activate radcalnet_oc

Install the package
-------------------

From PyPI:

.. code-block:: bash

   pip install radcalnet_oc

or from the root of a clone of the repository:

.. code-block:: bash

   pip install .

Look-up tables
--------------

The radiative transfer TOA look-up table (LUT) of the atmosphere, ``toa_lut_opac_wind_v3.nc`` (about
900 MB), is not part of the package. It is downloaded from
`Zenodo <https://doi.org/10.5281/zenodo.23394202>`__ (checksum verified) the first time the LUTs are
loaded, into the LUT folder: ``~/.local/share/radcalnet_oc/lut`` by default on Linux. It is the same LUT
as for `GRS <https://github.com/CNES/GRSprocessor>`__: a GRS LUT folder can be used as is.

.. code-block:: bash

   radcalnet_oc_config                        # print the configuration in use and check the LUT
   radcalnet_oc_config --download             # download the LUT now (e.g. before working offline)
   radcalnet_oc_config --init /path/to/lut    # use another LUT folder (e.g. the GRS one)

If the LUT is not available (no internet access, automatic download switched off), a light version
shipped with the package is used, with a lower accuracy, and a warning is logged.

The gaseous absorption LUTs, the solar spectra and the irradiance transmittance LUT are included in
the package (``radcalnet_oc/data``).

Configuration
-------------

The settings are read from, in increasing order of priority:

1. the default configuration of the package, ``radcalnet_oc/config.yml`` (not to be edited);
2. the user configuration file, ``~/.config/radcalnet_oc/config.yml`` (or the file given by the
   environment variable ``RADCALNET_OC_CONFIG``), created by ``radcalnet_oc_config --init``; it only
   needs the keys that differ from the defaults;
3. the environment variable ``RADCALNET_OC_DATA_ROOT``, which sets the LUT folder.

Default configuration:

.. code-block:: yaml

   path:
     lutdata: null        # LUT folder; null: ~/.local/share/radcalnet_oc/lut
     toa_lut: 'toa_lut_opac_wind_v3.nc'
     trans_lut: 'transmittance_lut_opac_wind_v3.nc'

   zenodo:
     url: 'https://zenodo.org'
     record: 23394202
     auto_download: true  # also switched off by RADCALNET_OC_NO_DOWNLOAD=1

   settings:
     aerosol_combination: [0, 0.5, 0, 0.5, 0., 0.]
     aerosol_models: ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70', 'WASO_rh0']

   processor:
     ncpu: 8
     chunk: 512
     netcdf_engine: 'netcdf4'

For instance, a user file pointing to an existing LUT folder and changing the default aerosol models
proportions:

.. code-block:: yaml

   path:
     lutdata: '/data/grs/grsdata'
   settings:
     aerosol_combination: [0, 0, 0, 1, 0, 0]

``aerosol_combination`` gives the default proportions of the ``aerosol_models`` (see
:ref:`methods-aerosol`). Run ``radcalnet_oc_config`` to print the resulting configuration.

Run the tests
-------------

.. code-block:: bash

   pip install ".[dev]"
   pytest

The tests use the light LUT of the package and never download the full LUT. The tests of the full LUT
run when it is in the LUT folder of the configuration (or given by the environment variable
``RADCALNET_OC_TEST_LUT``).

Build this documentation
------------------------

.. code-block:: bash

   pip install ".[docs]"
   sphinx-build -b html docs/source docs/build/html

The tutorial notebooks are not executed: they are rendered with the outputs saved in them.
