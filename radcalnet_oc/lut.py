'''
Look-up tables, auxiliary data (solar spectra, Rayleigh optical thickness) and spectral response of
the satellite sensors.

The paths of the atmosphere look-up tables are set in ``config.yml`` (see :doc:`/installation`);
the other data are installed with the package.
'''

import os

import numpy as np
import xarray as xr
import pandas as pd
import datetime

from numba import njit, prange

import matplotlib.pyplot as plt

import logging
from importlib.resources import files
import yaml

opj = os.path.join

# ------------------------------------
# get path of packaged files
# ------------------------------------
dir, filename = os.path.split(__file__)

thuillier_file = files('radcalnet_oc.data.auxdata').joinpath('ref_atlas_thuillier3.nc')
gueymard_file = files('radcalnet_oc.data.auxdata').joinpath('NewGuey2003.dat')
kurucz_file = files('radcalnet_oc.data.auxdata').joinpath('kurucz_0.1nm.dat')
tsis_file = files('radcalnet_oc.data.auxdata').joinpath(
    'hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc')
sunglint_eps_file = files('radcalnet_oc.data.auxdata').joinpath('mean_rglint_small_angles_vza_le_12_sza_le_60.txt')
rayleigh_file = files('radcalnet_oc.data.auxdata').joinpath('rayleigh_bodhaine.txt')
LUT_FILE_BACKUP = files('radcalnet_oc.data.lut.atmo').joinpath('toa_lut_opac_ultra_light.nc')

# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

LUTDATA = config['path']['lutdata']
TOALUT = config['path']['toa_lut']
TRANSLUT = config['path']['trans_lut']
TRANSLUT = files('radcalnet_oc.data.lut.atmo').joinpath(TRANSLUT)
CAMS_PATH = config['path']['trans_lut']
NCPU = config['processor']['ncpu']
NETCDF_ENGINE = config['processor']['netcdf_engine']

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']

# in-memory cache of the decompressed TOA radiance tables (see LUT.toa_lut), shared by the LUT objects
# (a Process creates a new LUT object at each run); one entry per aerosol model, wind speed and azimuth
# range, up to 90 MB each
TOA_LUT_CACHE = {}
TOA_LUT_CACHE_SIZE = 12


@njit(fastmath=True)
def Gamma2sigma(Gamma):
    r'''
    Standard deviation of a Gaussian function from its full width at half maximum,
    :math:`\sigma = \Gamma / (2\sqrt{2\ln 2})`.

    :param Gamma: full width at half maximum
    :return: standard deviation, in the unit of ``Gamma``
    '''
    return Gamma / (2. * np.sqrt(2. * np.log(2.)))


@njit(parallel=True, fastmath=True)
def gaussian(x, mu, sigma):
    r'''
    Normalized Gaussian function
    :math:`\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)`.

    :param x: abscissa (e.g. wavelengths in nm), 1-D array
    :param mu: mode of the Gaussian function
    :param sigma: standard deviation of the Gaussian function
    :return: values at ``x``, float32 array
    '''
    result = np.full((len(x)), np.nan, dtype=np.float32)
    for i in prange(len(result)):
        result[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x[i] - mu) ** 2 / (2 * sigma ** 2))
    return result


@njit(fastmath=True)
def super_gaussian(x,
                   amplitude=1.0,
                   mu=0.0,
                   sigma=1.0,
                   expon=2.0):
    r'''
    Super-Gaussian function, a Gaussian function with a flatter top for :math:`p > 2`:

    .. math::

       S(x) = \frac{A}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{|x-\mu|^{p}}{2\sigma^{p}}\right)

    :param x: abscissa (e.g. wavelengths in nm), float or array
    :param amplitude: amplitude :math:`A`
    :param mu: center :math:`\mu`
    :param sigma: width parameter :math:`\sigma` (see :py:func:`super_gaussian_fwhm2sigma`)
    :param expon: exponent :math:`p` (2 for a Gaussian function)
    :return: values at ``x``
    '''

    sigma = max(1.e-15, sigma)
    return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))


@njit(fastmath=True)
def super_gaussian_fwhm2sigma(fwhm,
                              expon):
    r'''
    Width parameter of a super-Gaussian function from its full width at half maximum,
    :math:`\sigma = \frac{\Gamma}{2}\,(2\ln 2)^{-1/p}`.

    :param fwhm: full width at half maximum :math:`\Gamma`
    :param expon: exponent :math:`p` of the super-Gaussian function
    :return: width parameter :math:`\sigma`, in the unit of ``fwhm``
    '''
    return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)


class LUT:
    '''
    Look-up tables of the atmosphere and of the gaseous absorption.

    At initialization, the tables are loaded (:py:meth:`load_auxiliary_data`); :py:meth:`lut_preparation`
    then combines the aerosol models and interpolates the tables at the geometry of the simulation.

    :ivar aero_lut: TOA look-up table (:py:class:`xarray.Dataset`): normalized radiance ``I`` and aerosol
        optical thickness ``aot`` for the aerosol models, wind speeds, geometries and reference aerosol
        optical thicknesses ``aot_ref`` (at 550 nm); wavelengths in nm
    :ivar trans_lut: irradiance transmittance look-up table (:py:class:`xarray.Dataset`)
    :ivar gas_lut: normalized absorption optical thickness of the gases
    :ivar naot_lut: aerosol optical thickness of each model normalized by ``aot_ref``, used by
        :py:class:`~radcalnet_oc.kernel.Aerosol`
    '''

    def __init__(self,
                 wl=np.arange(350, 2500, 10),
                 lut_file=opj(LUTDATA, TOALUT),
                 trans_lut_file=TRANSLUT):
        '''
        :param wl: wavelengths (nm) used by :py:meth:`lut_preparation_all_models` and for the auxiliary data
        :param lut_file: path of the TOA look-up table (by default ``lutdata/toa_lut`` of ``config.yml``);
            if not found, the light look-up table of the package is used
        :param trans_lut_file: path of the irradiance transmittance look-up table (by default the one of the
            package named ``trans_lut`` in ``config.yml``)
        '''

        # set parameters
        self.wl = wl

        # get path of necessary look-up tables

        self.lut_file = lut_file
        self.trans_lut_file = trans_lut_file
        self.dirdata = config['path']['lutdata']
        self.abs_gas_file = files('radcalnet_oc.data.lut.gases') / 'lut_abs_opt_thickness_normalized.nc'
        # self.lut_file = opj(self.dirdata, 'lut', 'opac_osoaa_lut_v2.nc')
        self.water_vapor_transmittance_file = files('radcalnet_oc.data.lut.gases') / 'water_vapor_transmittance.nc'

        self.load_auxiliary_data()

    def load_auxiliary_data(self):
        '''
        Load the look-up tables: TOA radiance, irradiance transmittance, gaseous absorption and water vapour
        transmittance.

        The wavelengths are converted from µm to nm. If the TOA look-up table is not found, the light
        version of the package is used, with a lower accuracy (logged at ``INFO`` level).
        '''

        logging.info('loading look-up tables')
        self.trans_lut = xr.open_dataset(self.trans_lut_file, engine=NETCDF_ENGINE)


        # convert wavelength in nanometer
        self.trans_lut['wl'] = self.trans_lut['wl'] * 1000
        self.trans_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        try:
            self.aero_lut = xr.open_dataset(self.lut_file, engine=NETCDF_ENGINE)
        except:
            logging.info('LUT file ' + self.lut_file + ' not found, please download and save it the proper directory')
            logging.info('LUT has been replaced with light dataset that might produce inaccuracies')
            self.aero_lut = xr.open_dataset(LUT_FILE_BACKUP, engine=NETCDF_ENGINE)

        # convert wavelength in nanometer
        self.aero_lut['wl'] = self.aero_lut['wl'] * 1000
        self.aero_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        self.aero_lut['aot'] = self.aero_lut.aot.isel(wind=0).squeeze()

        # get normalized aot for aerosol model fitting
        self.naot_lut = self.aero_lut.aot.interp(aot_ref=0.1) / 0.1

        self.gas_lut = xr.open_dataset(self.abs_gas_file, engine=NETCDF_ENGINE)
        self.Twv_lut = xr.open_dataset(self.water_vapor_transmittance_file, engine=NETCDF_ENGINE)

    def lut_preparation(self,
                        wind=2,
                        sza=[20, 40, 60],
                        vza=[0],
                        azi=[0],
                        aot_refs=np.linspace(0.0, 0.8, 25),
                        TLu_exponent =1.07,
                        aerosol_combination=AEROSOL_COMBINATION):
        r'''
        Combine the aerosol models and interpolate the look-up tables at the geometry of the simulation.

        The tables of the aerosol models are combined linearly with the proportions ``aerosol_combination``
        (:ref:`methods-aerosol`) and interpolated at the reference aerosol optical thicknesses ``aot_refs``.

        :param wind: wind speed (m s-1); the nearest wind speed of the tables is used
        :param sza: solar zenith angles (deg)
        :param vza: viewing zenith angles (deg)
        :param azi: relative azimuth angles (deg)
        :param aot_refs: reference aerosol optical thicknesses at 550 nm of the interpolated tables
        :param TLu_exponent: exponent :math:`\gamma` of the radiance transmittance,
            :math:`t^{\uparrow} = (T^{\uparrow})^{\gamma}`
        :param aerosol_combination: proportion of each aerosol model of ``config.yml``
            (``settings: aerosol_models``), list, array or :py:class:`xarray.DataArray`

        Sets the attributes:

        - ``trans_Ed``: irradiance transmittance :math:`T^{\downarrow}` for the solar zenith angles
        - ``trans_Eu``, ``trans_Lu``: upward irradiance and radiance transmittances for the viewing angles
        - ``Rdiff_lut``: atmospheric path reflectance :math:`R_{atm} = I/\mu_0`
        - ``Rray``: Rayleigh reflectance (no aerosol)
        - ``aot_lut``: spectral aerosol optical thickness of the mixture
        - ``rot``, ``sunglint_eps``: Rayleigh optical thickness and spectral shape of the sunglint
        '''
        logging.info('LUT preparation')

        if isinstance(aerosol_combination, (list, np.ndarray)):
            aerosol_combination = xr.DataArray(aerosol_combination, coords={'model': AEROSOL_MODELS})
        elif not isinstance(aerosol_combination, xr.DataArray):
            logging.error("error in aerosol_combination parameter, should be numpy or xr.DataArray")
            return

        self.aerosol_combination=aerosol_combination

        # the models of null proportion do not contribute to the mixture: they are not read nor interpolated
        models = aerosol_combination.model.values[aerosol_combination.values != 0]
        if len(models) == 0:
            models = aerosol_combination.model.values
        aerosol_combination = aerosol_combination.sel(model=models)

        aero_lut = self.aero_lut.sel(wind=wind, method='nearest')
        trans_lut = self.trans_lut.sel(wind=wind, method='nearest').sel(model=models)
        # TOA radiance of the Rayleigh atmosphere (first model, aot_ref=0) and of the aerosol models, in memory
        lut_models = [aero_lut.model.values[0], *[model for model in models if model != aero_lut.model.values[0]]]
        toa_lut = self.toa_lut(aero_lut.wind.values, lut_models, azi)

        # -----------------------------
        # interpolation transmittance
        # transmittance for irradiance and radiance
        # -----------------------------

        self.trans_Ed = trans_lut.interp(sza=sza)#.sortby('sza') #, *vza
        self.trans_Ed = (self.trans_Ed * aerosol_combination).sum('model')
        self.trans_Ed = self.trans_Ed.interp(aot_ref=aot_refs, method='quadratic')
        #self.trans_aero_lut = self.trans_aero_lut.interp(wl=self.wl, method='quadratic')
        # clean up the xarray DataArray object:
        self.trans_Ed = self.trans_Ed.to_dataarray().squeeze().reset_coords(drop=True)

        self.trans_Eu = trans_lut.interp(sza=vza).sortby('sza').rename({'sza': 'vza'})
        self.trans_Eu = (self.trans_Eu * aerosol_combination).sum('model')
        self.trans_Eu = self.trans_Eu.interp(aot_ref=aot_refs, method='quadratic')
        self.trans_Eu = self.trans_Eu.to_dataarray().squeeze().reset_coords(drop=True)
        self.trans_Lu = self.trans_Eu**TLu_exponent

        # -----------------------------
        # interpolation Rayleigh
        # -----------------------------
        # aot_ref=0 is a node of the table: selecting it is exact and avoids interpolating all the aot_ref
        self.Rray = self._interp_geometry(toa_lut.isel(model=0).sel(aot_ref=0), sza, vza, azi)
        self.Rray = self.Rray / np.cos(np.radians(self.Rray.sza))
        #self.Rray = self.Rray.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation atmo diffuse light
        # -----------------------------
        self.Rdiff_lut = self._interp_geometry(toa_lut.sel(model=models), sza, vza, azi)
        self.Rdiff_lut = (self.Rdiff_lut * aerosol_combination).sum('model')
        self.Rdiff_lut = self.Rdiff_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.Rdiff_lut = self.Rdiff_lut / np.cos(np.radians(self.Rdiff_lut.sza))
        #self.Rdiff_lut = self.Rdiff_lut.interp(wl=self.wl, method='quadratic')

        self.aot_lut = (aero_lut.aot.sel(model=models) * aerosol_combination).sum('model')
        self.aot_lut = self.aot_lut.interp(aot_ref=aot_refs,
                                           method='quadratic'
                                           )#.interp(wl=self.wl, method='quadratic')

        self.szas = self.Rdiff_lut.sza.values
        self.vzas = self.Rdiff_lut.vza.values
        self.azis = self.Rdiff_lut.azi.values
        self.aot_refs = self.Rdiff_lut.aot_ref.values

        _auxdata = AuxData(wl=self.wl)  # wl=masked.wl)
        self.sunglint_eps = _auxdata.sunglint_eps  # ['mean'].interp(wl=wl)
        self.rot = _auxdata.rot

    def toa_lut(self,
                wind,
                models,
                azi):
        '''
        Normalized TOA radiance ``I`` of the look-up table for a wind speed and some aerosol models, loaded
        in memory for the azimuth range bracketing ``azi``.

        Reading the compressed table is the most expensive step of :py:meth:`lut_preparation`: the tables
        read are kept in a cache shared by all the :py:class:`LUT` objects (``TOA_LUT_CACHE``, at most
        ``TOA_LUT_CACHE_SIZE`` entries), so that the next preparations with the same file, wind speed,
        models and azimuth range do not read the file again.

        :param wind: wind speed (m s-1), a value of the ``wind`` coordinate of the table
        :param models: names of the aerosol models
        :param azi: relative azimuth angles (deg) to be interpolated
        :return: :py:class:`xarray.DataArray` with a ``model`` dimension, without the ``wind`` dimension
        '''
        azis = self.aero_lut.azi.values
        azi = np.atleast_1d(azi)
        i0 = max(int(np.searchsorted(azis, np.min(azi), side='right')) - 1, 0)
        i1 = min(int(np.searchsorted(azis, np.max(azi), side='left')) + 1, len(azis))

        keys = {model: (str(self.lut_file), float(wind), str(model), i0, i1) for model in models}
        missing = [model for model in models if keys[model] not in TOA_LUT_CACHE]
        if missing:
            # read the missing models at once: they can share the chunks of the compressed file
            lut = self.aero_lut.I.sel(wind=wind, model=missing).isel(azi=slice(i0, i1)).load()
            for model in missing:
                if len(TOA_LUT_CACHE) >= TOA_LUT_CACHE_SIZE:
                    # remove the oldest entry
                    TOA_LUT_CACHE.pop(next(iter(TOA_LUT_CACHE)))
                # copy: the entry must not keep the array of all the models alive
                TOA_LUT_CACHE[keys[model]] = lut.sel(model=model).copy()

        return xr.concat([TOA_LUT_CACHE[keys[model]] for model in models], dim='model')

    @staticmethod
    def _interp_geometry(lut, sza, vza, azi):
        '''
        Linear interpolation of a look-up table at the solar zenith, viewing zenith and relative azimuth
        angles.

        The multilinear interpolation is separable: the dimensions are interpolated one after the other,
        starting with those reduced the most (azimuth and viewing angles), so that only the table nodes
        bracketing the requested angles are read and the next interpolations work on smaller arrays.

        :param lut: :py:class:`xarray.DataArray` with ``sza``, ``vza`` and ``azi`` dimensions
        :param sza: solar zenith angles (deg)
        :param vza: viewing zenith angles (deg)
        :param azi: relative azimuth angles (deg)
        :return: interpolated :py:class:`xarray.DataArray`
        '''
        return lut.interp(azi=azi).interp(vza=vza).interp(sza=sza)

    def lut_preparation_all_models(self,
                                   wind=2,
                                   sza=[20, 40, 60],
                                   vza=[0],
                                   azi=[0],
                                   aot_refs=np.linspace(0.0, 0.8, 25),
                                   ):
        '''
        Interpolate the look-up tables at the geometry of the simulation for each aerosol model, without
        combining them (the ``model`` dimension is kept), at the wavelengths ``wl`` of the object.

        :param wind: wind speed (m s-1); the nearest wind speed of the tables is used
        :param sza: solar zenith angles (deg)
        :param vza: viewing zenith angles (deg)
        :param azi: relative azimuth angles (deg)
        :param aot_refs: reference aerosol optical thicknesses at 550 nm of the interpolated tables

        Sets the attributes ``trans_aero_lut``, ``Rray``, ``Rdiff_lut``, ``aot_lut``, ``rot`` and
        ``sunglint_eps``.
        '''
        logging.info('LUT preparation')

        aero_lut = self.aero_lut.sel(wind=wind, method='nearest')
        trans_lut = self.trans_lut.sel(wind=wind, method='nearest')

        # -----------------------------
        # interpolation transmittance
        # -----------------------------
        self.trans_aero_lut = trans_lut.interp(sza=[*sza, *vza])
        self.trans_aero_lut = self.trans_aero_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.trans_aero_lut = self.trans_aero_lut.interp(wl=self.wl, method='quadratic')
        # clean up the xarray DataArray object:
        self.trans_aero_lut = self.trans_aero_lut.to_dataarray().squeeze().reset_coords(drop=True)

        # -----------------------------
        # interpolation Rayleigh
        # -----------------------------
        toa_lut = self.toa_lut(aero_lut.wind.values, aero_lut.model.values, azi)
        self.Rray = self._interp_geometry(toa_lut.sel(aot_ref=0), sza, vza, azi)
        self.Rray = self.Rray / np.cos(np.radians(self.Rray.sza))
        self.Rray = self.Rray.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation atmo diffuse light
        # -----------------------------
        self.Rdiff_lut = self._interp_geometry(toa_lut, sza, vza, azi)
        self.Rdiff_lut = self.Rdiff_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.Rdiff_lut = self.Rdiff_lut / np.cos(np.radians(self.Rdiff_lut.sza))
        self.Rdiff_lut = self.Rdiff_lut.interp(wl=self.wl, method='quadratic')

        self.aot_lut = aero_lut.aot.interp(aot_ref=aot_refs, method='quadratic').interp(wl=self.wl, method='quadratic')

        self.szas = self.Rdiff_lut.sza.values
        self.vzas = self.Rdiff_lut.vza.values
        self.azis = self.Rdiff_lut.azi.values
        self.aot_refs = self.Rdiff_lut.aot_ref.values

        _auxdata = AuxData(wl=self.wl)  # wl=masked.wl)
        self.sunglint_eps = _auxdata.sunglint_eps  # ['mean'].interp(wl=wl)
        self.rot = _auxdata.rot


class AuxData():
    '''
    Auxiliary spectral data: Rayleigh optical thickness and mean spectral shape of the sunglint
    reflectance.

    :ivar rot: Rayleigh optical thickness (Bodhaine et al., 1999) for 1013.25 hPa
    :ivar sunglint_eps: mean spectral shape of the sunglint reflectance (small angles,
        vza <= 12 deg, sza <= 60 deg)
    :ivar pressure_rot_ref: reference pressure of ``rot`` (hPa)
    '''

    def __init__(self,
                 wl=None):
        # load data from raw files
        # self.solar_irr = SolarIrradiance()
        '''
        :param wl: wavelengths (nm) at which ``rot`` and ``sunglint_eps`` are interpolated; if None, the
            original wavelengths are kept
        '''
        self.sunglint_eps = pd.read_csv(sunglint_eps_file, sep=r'\s+', index_col=0).to_xarray()
        self.rayleigh()
        self.pressure_rot_ref = 1013.25

        # reproject onto desired wavelengths
        if wl is not None:
            # self.solar_irr = self.solar_irr.interp(wl=wl)
            self.sunglint_eps = self.sunglint_eps['mean'].interp(wl=wl)
            self.rot = self.rot.interp(wl=wl)

    def rayleigh(self):
        '''
        Load the Rayleigh optical thickness for P = 1013.25 hPa, T = 288.15 K and 360 ppm of CO2, from
        Bodhaine, B. A., Wood, N. B., Dutton, E. G., Slusser, J. R. (1999). On Rayleigh optical depth
        calculations, J. Atmos. Ocean. Tech., 16, 1854-1861.

        Sets the attribute ``rot`` (:py:class:`xarray.DataArray`, wavelengths in nm).
        '''
        data = pd.read_csv(rayleigh_file, skiprows=16, sep=' ', header=None)
        data.columns = ('wl', 'rot', 'dpol')
        self.rot = data.set_index('wl').to_xarray().rot
        self.rot.attrs['description']="Rayleigh Optical Thickness for P=1013.25mb, T=288.15K, CO2=360ppm"
        self.rot.attrs['reference'] = "Bodhaine, B.A., Wood, N.B, Dutton, E.G., Slusser, J.R. (1999). On Rayleigh " + \
                                      "Optical Depth Calculations, J. Atmos. Ocean Tech., 16, 1854-1861."


class SolarIrradiance():
    '''
    Extraterrestrial solar irradiance spectra at mean Earth-Sun distance, in mW m-2 nm-1, between 300 and
    2600 nm.

    :ivar tsis: TSIS-1 hybrid solar reference spectrum (Coddington et al., 2021), 0.1 nm resolution
    :ivar thuillier: Thuillier et al. (2003) spectrum
    :ivar gueymard: Gueymard (2004) spectrum
    :ivar kurucz: Kurucz (1992) spectrum, 0.1 nm resolution
    '''

    def __init__(self, wl=None):
        # load data from raw files
        '''
        :param wl: not used
        '''
        self.wl_min = 300
        self.wl_max = 2600

        self.gueymard = self.read_gueymard()
        self.kurucz = self.read_kurucz()
        self.thuillier = self.read_thuillier()
        self.tsis = self.read_tsis()

    def read_tsis(self):
        '''
        Read the TSIS-1 hybrid solar reference spectrum.

        :return: solar irradiance (mW m-2 nm-1), :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        '''
        tsis = xr.open_dataset(tsis_file)
        tsis = tsis.set_index(wavelength='Vacuum Wavelength').rename(
            {'wavelength': 'wl'})  # set_coords('Vacuum Wavelength')
        # convert
        tsis['SSI'] = tsis.SSI * 1000  # .plot(lw=0.5)
        tsis.SSI.attrs['units'] = 'mW m-2 nm-1'
        tsis.SSI.attrs['long_name'] = 'Solar Spectral Irradiance Reference Spectrum (mW m-2 nm-1)'
        tsis.SSI.attrs['reference'] = 'Coddington, O. M., Richard, E. C., Harber, D., et al. (2021).' + \
                                      'The TSIS-1 Hybrid Solar Reference Spectrum. Geophysical Research Letters,' + \
                                      '48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709'
        return tsis.SSI.sel(wl=slice(self.wl_min, self.wl_max))

    def read_thuillier(self):
        '''
        Read the Thuillier solar spectrum.

        :return: solar irradiance (mW m-2 nm-1), :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        '''
        solar_irr = xr.open_dataset(thuillier_file).squeeze().data.drop('time') * 1e3
        solar_irr = solar_irr.rename({'wavelength': 'wl'})
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        return solar_irr

    def read_gueymard(self):
        '''
        Read the Gueymard solar spectrum.

        :return: solar irradiance (mW m-2 nm-1), :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        '''
        solar_irr = pd.read_csv(gueymard_file, sep=r'\s+', skiprows=30, header=None)
        solar_irr.columns = ['wl', 'data']
        solar_irr = solar_irr.set_index('wl').data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        solar_irr.attrs['reference'] = 'Gueymard, C. A., Solar Energy, Volume 76, Issue 4,2004, ISSN 0038-092X'
        return solar_irr

    def read_kurucz(self):
        '''
        Read the Kurucz solar spectrum.

        :return: solar irradiance (mW m-2 nm-1), :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        '''
        solar_irr = pd.read_csv(kurucz_file, sep=r'\s+', skiprows=11, header=None)
        solar_irr.columns = ['wl', 'data']
        solar_irr = solar_irr.set_index('wl').data.to_xarray()
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        solar_irr.attrs['reference'] = 'Kurucz, R.L., Synthetic infrared spectra, in Infrared Solar Physics, ' + \
                                       'IAU Symp. 154, edited by D.M. Rabin and J.T. Jefferies, Kluwer, Acad., ' + \
                                       'Norwell, MA, 1992.'
        return solar_irr

    def interp(self, wl=[440, 550, 660, 770, 880]):
        '''
        Interpolate the Thuillier and Gueymard spectra at new wavelengths (in place).

        :param wl: wavelengths (nm)
        '''
        self.thuillier = self.thuillier.interp(wl=wl)
        self.gueymard = self.gueymard.interp(wl=wl)


class Spectral():
    '''
    Spectral response of the bands of a satellite sensor, modelled from their central wavelength and
    full width at half maximum (FWHM), and convolution of hyperspectral signals with these responses
    (see :ref:`spectral_convolution`).

    Example::

        spectral = Spectral(central_wl=np.array([443., 490., 560., 665., 865.]), fwhm=20.)
        Rtoa_bands = spectral.convolve2(radcalnet_db.Rtoa)

    :ivar fwhm: FWHM of the bands (nm), :py:class:`xarray.DataArray` with the central wavelengths as
        ``wl`` coordinate
    '''

    def __init__(self,
                 central_wl,
                 fwhm):
        '''
        :param central_wl: central wavelengths of the bands (nm), numpy array
        :param fwhm: full width at half maximum of the bands (nm), scalar (same for all the bands) or numpy
            array
        '''
        self.central_wl = central_wl
        if not isinstance(fwhm, np.ndarray):
            fwhm = np.array([fwhm] * len(central_wl))
        fwhm = xr.DataArray(fwhm, name='fwhm',
                            coords={'wl': central_wl},
                            attrs={
                                'definition': 'full width at half maximum of spectral responses modeled as gaussian distributions'})
        self.fwhm = fwhm

    def plot_rsr(self,
                 expon=None):
        '''
        Plot the spectral response functions of the bands, normalized to a maximum of 1.

        :param expon: None for the Gaussian responses used by :py:meth:`convolve`, or the exponent of
            the super-Gaussian responses used by :py:meth:`convolve2` (3 by default in ``convolve2``)
        :return: :py:class:`matplotlib.figure.Figure`
        '''
        wl_ref = np.linspace(350, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in zip(self.fwhm.wl.values, self.fwhm.values):
            if expon is None:
                rsr = gaussian(wl_ref, float(mu), Gamma2sigma(float(fwhm)))
            else:
                rsr = super_gaussian(wl_ref, mu=float(mu), sigma=super_gaussian_fwhm2sigma(float(fwhm), expon),
                                     expon=expon)
            axs.plot(wl_ref, rsr / rsr.max(), '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Relative spectral response')

        return fig

    @staticmethod
    @njit(parallel=True)
    def convolve_(
            wl_signal,
            signal,
            wl,
            fwhm,
    ):
        '''
        Convolution of a spectrum with Gaussian spectral responses (numba function).

        :param wl_signal: wavelengths of the signal (nm), 1-D array
        :param signal: spectrum to convolve, 1-D array at ``wl_signal``
        :param wl: central wavelengths of the bands (nm)
        :param fwhm: full width at half maximum of the bands (nm)
        :return: band values, float32 array
        '''
        Nwl = len(wl)
        signal_ = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = Gamma2sigma(fwhm[ii])
            rsr = gaussian(wl_signal, wl[ii], sig)
            signal_[ii] = np.trapezoid((signal * rsr), wl_signal) / np.trapezoid(rsr, wl_signal)
        return signal_

    @staticmethod
    @njit(parallel=True)
    def convolve2_(
            wl_signal,
            signal,
            wl,
            fwhm,
            expon=2.,
            threshold=1e-6
    ):
        '''
        Convolution of a spectrum with super-Gaussian spectral responses (numba function).

        :param wl_signal: wavelengths of the signal (nm), 1-D array
        :param signal: spectrum to convolve, 1-D array at ``wl_signal``
        :param wl: central wavelengths of the bands (nm)
        :param fwhm: full width at half maximum of the bands (nm)
        :param expon: exponent of the super-Gaussian function (2 for a Gaussian function)
        :param threshold: values of the spectral response below this threshold are ignored, to speed up the
            computation
        :return: band values, float32 array
        '''

        Nwl = len(wl)
        response = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = super_gaussian_fwhm2sigma(fwhm[ii], expon)
            rsr = super_gaussian(wl_signal, mu=wl[ii], sigma=sig, expon=expon)

            # remove values above a given threshold to speed up computation
            idx = rsr > threshold
            wl_signal_ =wl_signal[idx]
            signal_ = signal[idx]
            rsr = rsr[idx]

            response[ii] = np.trapezoid((signal_ * rsr), wl_signal_) / np.trapezoid(rsr, wl_signal_)
        return response

    def band_weights(self,
                     wl_signal,
                     expon=None,
                     threshold=None):
        '''
        Integration weights of the spectral response of each band on the wavelengths of the signal.

        The convolution of Eq. :eq:`convolution`, computed with the trapezoidal rule, is linear in the
        signal: the band value is the dot product of the signal with weights that depend only on the
        wavelengths. The weights are computed once and applied to all the spectra
        (:py:meth:`convolve_nd`), which gives the same results as :py:meth:`convolve_` and
        :py:meth:`convolve2_`.

        :param wl_signal: wavelengths of the signal (nm), 1-D array
        :param expon: None for the Gaussian responses of :py:meth:`convolve`, or the exponent of the
            super-Gaussian responses of :py:meth:`convolve2`
        :param threshold: values of the spectral response below this threshold are ignored (super-Gaussian
            responses only)
        :return: list of ``(start, stop, weights)`` per band: the band value is
            ``signal[start:stop] @ weights``; ``weights`` is None when the response does not overlap the
            signal (band value NaN)
        '''
        wl_signal = np.asarray(wl_signal, dtype=np.float64)
        bands = []
        for mu, fwhm in zip(self.fwhm.wl.values, self.fwhm.values):
            if expon is None:
                rsr = gaussian(wl_signal, float(mu), Gamma2sigma(float(fwhm))).astype(np.float64)
                start, stop = 0, len(wl_signal)
            else:
                rsr = super_gaussian(wl_signal, mu=float(mu),
                                     sigma=super_gaussian_fwhm2sigma(float(fwhm), expon), expon=expon)
                idx = np.flatnonzero(rsr > threshold)
                if len(idx) == 0:
                    bands.append((0, 0, None))
                    continue
                # the response is unimodal: the wavelengths above the threshold are contiguous
                start, stop = idx[0], idx[-1] + 1
            rsr = rsr[start:stop]
            # trapezoidal rule coefficients
            dwl = np.diff(wl_signal[start:stop]) / 2
            coef = np.zeros_like(rsr)
            coef[:-1] += dwl
            coef[1:] += dwl
            norm = np.sum(rsr * coef)
            bands.append((start, stop, rsr * coef / norm if norm > 0 else None))
        return bands

    def convolve_nd(self,
                    signal,
                    bands):
        '''
        Apply the band weights of :py:meth:`band_weights` to each spectrum of a multidimensional signal.

        The computation is vectorized over all the spectra; for a dask-backed signal it is lazy and done
        chunk by chunk (the ``wl`` dimension is gathered in a single chunk).

        :param signal: spectral signal, :py:class:`xarray.DataArray` with a ``wl`` dimension (nm) and any
            other dimensions
        :param bands: band weights returned by :py:meth:`band_weights`
        :return: :py:class:`xarray.DataArray` with the central wavelengths of the bands as ``wl`` coordinate
        '''

        def apply_weights(sig):
            # sig: numpy array with the wavelengths on the last axis
            out = np.full(sig.shape[:-1] + (len(bands),), np.nan, dtype=np.float32)
            for ii, (start, stop, weights) in enumerate(bands):
                if weights is not None:
                    out[..., ii] = sig[..., start:stop] @ weights
            return out

        if signal.chunks is not None:
            signal = signal.chunk({'wl': -1})
        signal_int = xr.apply_ufunc(
            apply_weights,
            signal,
            input_core_dims=[['wl']],
            output_core_dims=[['wl_band']],
            dask='parallelized',
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'output_sizes': {'wl_band': len(bands)}},
        )
        return signal_int.rename(wl_band='wl').assign_coords(wl=self.fwhm.wl.values)

    def convolve2(self,
                  signal,
                  name='signal',
                  expon=3,
                  threshold=1e-4,
                  info={}):
        '''
        Convolve a signal with super-Gaussian spectral responses of the bands, Eq. :eq:`convolution`.

        :param signal: spectral signal, :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        :param name: name of the output variable if ``signal`` has no name
        :param expon: exponent of the super-Gaussian function (2 for a Gaussian function)
        :param threshold: values of the spectral response below this threshold are ignored, to speed up the
            computation
        :param info: not used (the attributes of ``signal`` are kept)
        :return: band values: :py:class:`xarray.DataArray` for a 1-D signal, :py:class:`xarray.Dataset` with
            one variable for a multidimensional signal; the ``wl`` coordinate gives the central wavelengths
        '''

        attrs=signal.attrs
        name=signal.name if signal.name is not None else name
        bands = self.band_weights(signal.wl.values, expon=expon, threshold=threshold)
        if signal.ndim == 1:
            signal_int = self.convolve_nd(signal.reset_coords(drop=True), bands).rename(name)
        else:
            # to handle multidimensional xarray: convolution of each spectrum
            signal_int = self.convolve_nd(signal, bands).to_dataset(name=name)
        signal_int.attrs = attrs

        return signal_int

    def convolve(self,
                 signal,
                 name='signal',
                 info={}):
        '''
        Convolve a signal with Gaussian spectral responses of the bands, Eq. :eq:`convolution`.

        :param signal: spectral signal, :py:class:`xarray.DataArray` with a ``wl`` dimension (nm)
        :param name: name of the output variable
        :param info: attributes of the output
        :return: band values, :py:class:`xarray.DataArray` (with an extra ``variable`` dimension of size 1
            for a multidimensional signal); the ``wl`` coordinate gives the central wavelengths
        '''

        bands = self.band_weights(signal.wl.values)
        if signal.ndim == 1:
            signal_int = self.convolve_nd(signal.reset_coords(drop=True), bands).rename(name)
        else:
            # to handle multidimensional xarray: convolution of each spectrum
            signal_int = self.convolve_nd(signal, bands).to_dataset(name=name).to_dataarray()
        signal_int.attrs = info

        return signal_int
