'''RadCalNet-OC daily products of an AERONET-OC site: simulated TOA reflectance at nadir (10 nm), with the direct
sunglint and its flags, and a copy without the glinted times.

For each AERONET-OC measurement of the site (``aeronet_oc_db``, level ``--level``, from ``--start`` to ``--end``):
hyperspectral Rrs at nadir reconstructed with invRrs (``radcalnet_oc.reconstruction``) from Rrs_Lwn, aerosol mixture
retrieved from the spectral AOT (``radcalnet_oc.Aerosol``), simulation at nadir (``radcalnet_oc.Process``, settings
of the installed version), direct sunglint at the wind of the station (``Process.compute_sunglint``, slope statistics
``--glint-stats``). The simulation is run per month (restart: months already written are skipped), then written as
one file per day, convolved with super-Gaussian responses of 10 nm FWHM every 10 nm (350-2490 nm, exponent 3), as the
RadCalNet-OC products before version 1.2, in the release folder ``--odir`` (default
/data/radcalnet/products/v<radcalnet_oc version>; the monthly full-spectral simulations are cached in ``--cache``,
default /data/radcalnet/simulations/v<version>/monthly):

    <odir>/daily/<site>/<yyyy>/<mm>/<dd>/<site>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc

Variables: as before (Rtoa, Ed, E0, Rrs, sza, aerosol and gas inputs, aerosol_combination) and, new in version 1.2,
``wind``, ``Rglint`` (TOA direct glint at nadir, not included in ``Rtoa``), ``Rtoa_glint`` (``Rtoa + Rglint``),
``Rglint_unc`` (spread of the plausible slope statistics, wind +/- 1.5 m s-1), ``glint_uncertain`` (per band,
``Rglint_unc / Rtoa_glint`` > 5 %), ``glint_fraction`` (865 nm), ``glint_angle`` and ``glint_flags`` (bitmask
``radcalnet_oc.GLINT_FLAGS``: low >= 1 %, moderate >= 10 %, high >= 50 % of the TOA reflectance at 865 nm, glint angle
< 30 deg). The copy ``<odir>/daily_glint_filtered/<site>/...`` keeps only the times without the bits of
``--filter-flags`` (moderate and near-specular by default); with ``--filter-uncertain``, also without
``glint_uncertain`` in 400-900 nm (strict: at nadir, this removes most of the summer times of the platforms).

The chain is the module ``radcalnet_oc.products`` of the package (command ``radcalnet_oc-products``, same options);
this script keeps the defaults of the release folders of the project (/data/radcalnet).

Usage:
    python make_products.py --site AAOT --level lev20 --name VEIT
    python make_products.py --site Section-7_Platform --level lev15 --start 2024-08-19 --name CORO
'''
import radcalnet_oc as radoc
from radcalnet_oc import products
# the functions of the chain, for the scripts that import them from here (statement_figures.py)
from radcalnet_oc.products import (CENTRAL_WL, FWHM, EXPON, INPUT_VARS, UNC_RANGE, daily_file,  # noqa: F401
                                   glint_ok, monthly_file, simulate_month, to_10nm)

ROOT = '/data/radcalnet'   # layout: README.md of the folder
ODIR = f'{ROOT}/products/v{radoc.__version__}'
CACHE = f'{ROOT}/simulations/v{radoc.__version__}/monthly'


def main(argv=None):
    return products.main(argv, odir=ODIR, cache=CACHE, description=__doc__, prog='make_products.py')


if __name__ == '__main__':
    raise SystemExit(main())
