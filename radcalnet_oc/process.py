'''
Simulation of the top-of-atmosphere signal from surface measurements.

:py:class:`Process` chains the steps described in :doc:`/methods`: solar irradiance, gaseous and
atmospheric transmittances, downwelling irradiance, water-leaving radiance, atmospheric path
reflectance and top-of-atmosphere reflectance.
'''

import functools
import os
import time
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

import logging

from importlib.resources import files

from . import GaseousTransmittance, GasCoupling, Misc, LUT, SolarIrradiance, __version__
from .kernel import GAS_COUPLING, GAS_COUPLINGS, GAS_COUPLING_FILE, GAS_COUPLING_PPLD_FILE
from .config import load_config, LUT_FILE_BACKUP
from .lut import WIND, interp_wl_loglog
#: pressure of the Rayleigh optical thickness of the OSOAA runs of the look-up tables (hPa): AP.MOT of the runs is the
#: Bodhaine et al. (1999) optical thickness at 1015.2 hPa (e.g. 0.360251 at 400 nm against 0.359559 at 1013.25 hPa)
LUT_PRESSURE = 1015.2
from .water import extend_rrs
from . import coxmunk

logger = logging.getLogger(__name__)

template = files('radcalnet_oc.data.template').joinpath('template_clear_water.nc')
# --------------------------------------------------
# settings of the configuration (see radcalnet_oc.config)
# --------------------------------------------------
config = load_config()

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']
# spectral extension of Rrs beyond the range of the input: 'pure_water' or 'zero'
RRS_EXTENSIONS = ('pure_water', 'zero')
RRS_EXTENSION = config['settings'].get('rrs_extension', 'pure_water')
RAYLEIGH_PRESSURE = bool(config['settings'].get('rayleigh_pressure', True))


@functools.lru_cache(maxsize=None)
def package_version():
    '''
    Version of the package, with the git commit for a development installation (e.g.
    ``1.0.2+g6439281.dirty`` for an editable install with uncommitted changes); the plain version for an
    installed package.
    '''
    root = Path(__file__).resolve().parent.parent
    if not (root / '.git').exists():
        return __version__
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip()
        commit = git('rev-parse', '--short', 'HEAD')
        dirty = '.dirty' if git('status', '--porcelain', '--untracked-files=no', '--', 'radcalnet_oc') else ''
    except (OSError, subprocess.SubprocessError):
        return __version__
    return f'{__version__}+g{commit}{dirty}'


class Process():
    '''
    Simulation of the top-of-atmosphere (TOA) signal for a time series of surface measurements.

    The input dataset is read at initialization; :py:meth:`execute` computes the simulation and stores
    it in ``radcalnet_db``.

    Example::

        process = Process(input_db=input_db, vza=vza, azi=azi)
        process.execute()
        process.radcalnet_db.Rtoa

    :ivar lut: :py:class:`~radcalnet_oc.lut.LUT` object with the look-up tables
    :ivar full_wl: wavelengths of the simulation (nm), from the gaseous absorption LUT (350--2500 nm)
    :ivar F0: extraterrestrial solar irradiance at ``full_wl`` (mW m-2 nm-1)
    :ivar gas_trans: :py:class:`~radcalnet_oc.kernel.GaseousTransmittance` object
    :ivar radcalnet_db: output :py:class:`xarray.Dataset`, set by :py:meth:`execute`
    '''

    def __init__(self,

                 input_db=None,
                 vza=xr.DataArray([0], coords={'vza': [0]}, dims='vza'),
                 azi=xr.DataArray([0], coords={'azi': [0]}, dims='azi'),
                 central_wl=np.arange(350, 2500, 1),
                 solar_database='tsis',
                 Rrs_name='Rrs',
                 o2_model=None,
                 gas_coupling=None,
                 rrs_extension=None,
                 rayleigh_pressure=None
                 ):
        '''
        :param input_db: time series of the surface measurements, :py:class:`xarray.Dataset` with a
            ``time`` dimension and the variables listed in :doc:`/usage` (``Rrs``, ``sza``,
            ``day_of_year``, ``aot550``, ``pressure``, ``tcwv``, ``tco3``, ``tcno2``...);
            if None, the clear-water template of the package is used
        :param vza: viewing zenith angles (deg), :py:class:`xarray.DataArray` with a ``vza`` dimension
        :param azi: relative azimuth angles (deg), :py:class:`xarray.DataArray` with an ``azi`` dimension;
            0 deg when the viewing direction points towards the Sun (sunglint geometry), 180 deg for the
            backscattering geometry (radiative transfer convention of the look-up tables)
        :param central_wl: not used (the wavelengths are those of the gaseous absorption LUT)
        :param solar_database: solar spectrum used for :math:`F_0`: ``'tsis'`` (default), ``'thuillier'``,
            ``'gueymard'`` or ``'kurucz'``
        :param Rrs_name: name of the remote-sensing reflectance variable in ``input_db``
        :param o2_model: absorption of O2, ``'reptran'`` or ``'lbl'`` (line by line in the O2-B and O2-A
            bands, see :py:class:`~radcalnet_oc.kernel.GaseousTransmittance`); by default
            ``settings: o2_model`` of the configuration (``'lbl'``)
        :param gas_coupling: gaseous transmittance of the scattered light: ``'decoupled'`` (air mass of the
            direct beam, :math:`1/\mu_0 + 1/\mu_v` for ``Ratm`` and the water-leaving term, :math:`1/\mu_0` for
            ``Ed``), ``'effective_air_mass'`` (effective air masses or single-scattering blend per absorption
            window, computed at 5 cm-1) or ``'path_length'`` (photon path-length model, channel by channel),
            in the O2 and H2O bands of 684-990 nm, see :py:class:`~radcalnet_oc.kernel.GasCoupling`; by
            default ``settings: gas_coupling`` of the configuration (``'path_length'``)
        :param rrs_extension: remote-sensing reflectance beyond the spectral range of the input:
            ``'pure_water'`` (semi-analytical model with the absorption of pure water, fitted on the
            near-infrared end of the input, see :py:func:`~radcalnet_oc.water.extend_rrs`) or ``'zero'``
            (as before version 1.1); by default ``settings: rrs_extension`` of the configuration
            (``'pure_water'``)
        :param rayleigh_pressure: scale the Rayleigh scattering with the surface pressure of each measurement
            (the look-up tables are computed for 1015.2 hPa): path reflectance and diffuse transmittances, see
            :py:meth:`rayleigh_pressure_correction`; by default ``settings: rayleigh_pressure`` of the
            configuration (True)
        '''
        self.rayleigh_pressure = RAYLEIGH_PRESSURE if rayleigh_pressure is None else bool(rayleigh_pressure)
        self.rrs_extension = RRS_EXTENSION if rrs_extension is None else rrs_extension
        if self.rrs_extension not in RRS_EXTENSIONS:
            raise ValueError(f'rrs_extension must be one of {RRS_EXTENSIONS}, not {self.rrs_extension!r}')

        if input_db is None:
            input_db = xr.open_dataset(template)
        self.solar_database = solar_database

        self.input_db = input_db

        self.vza = vza
        self.azi = azi

        # get LUT object
        logger.debug('get LUT object')
        lut = LUT()
        self.lut = lut

        # get full spectral resolution wavelength from gas LUT
        # crop to 350 - 2500 range to comply with OSOAA lut
        full_wl = lut.gas_lut.wl.sel(wl=slice(350, 2500))
        self.full_wl = full_wl
        # lut.wl = full_wl

        logger.debug('get auxiliary data')
        lut.load_auxiliary_data()

        logger.debug('get solar irradiance')
        solar_irr = SolarIrradiance()
        self.F0 = solar_irr.__dict__[solar_database].interp(wl=full_wl)

        logger.debug('get gaseous transmittance object')
        self.gas_trans = GaseousTransmittance(lut.gas_lut, o2_model=o2_model)

        self.gas_coupling = GAS_COUPLING if gas_coupling is None else gas_coupling
        if self.gas_coupling not in GAS_COUPLINGS:
            raise ValueError(f'gas_coupling must be one of {GAS_COUPLINGS}, not {self.gas_coupling!r}')
        tables = {'effective_air_mass': GAS_COUPLING_FILE, 'path_length': GAS_COUPLING_PPLD_FILE}
        self.coupling = GasCoupling(tables[self.gas_coupling]) if self.gas_coupling in tables else None
        self.gas_coupling_file = tables.get(self.gas_coupling)

        logger.debug('set input parameters')
        self.set_param(Rrs_name=Rrs_name)

    def set_param(self,
                  Rrs_name='Rrs'):
        '''
        Read the solar geometry, the remote-sensing reflectance, the aerosol optical thickness and the
        Earth-Sun distance correction from the input dataset.

        :param Rrs_name: name of the remote-sensing reflectance variable in ``input_db``

        The wind speed (``wind``, m s-1, optional) sets the sea-surface roughness of the sky and Sun
        reflection; without it, the wind speed of ``config.yml`` (``settings: wind``, 2 m s-1) is used.

        Sets the attributes ``sza``, ``mu0``, ``muv``, ``Rrs``, ``aot550``, ``wind`` and ``D2``.
        '''

        input_db = self.input_db
        self.sza = input_db.sza
        self.mu0 = np.cos(np.radians(self.sza))
        self.muv = np.cos(np.radians(self.vza))

        self.Rrs = input_db[Rrs_name]
        # self.Rrs = self.Rrs.rename({"hyper_wl":"wl"})
        self.aot550 = input_db['aot550']
        if 'wind' in input_db:
            self.wind = input_db['wind']
            if self.wind.isnull().any():
                logger.warning(f'wind speed missing for {int(self.wind.isnull().sum())} measurement(s): '
                               f'{WIND:g} m s-1 used')
                self.wind = self.wind.fillna(WIND)
        else:
            self.wind = xr.full_like(self.sza, WIND, dtype=float).rename('wind')
            self.wind.attrs = {}

        # get correction for Sun-Earth distance
        self.D2 = Misc.earth_sun_correction(input_db['day_of_year'])

    def get_gas_transmittance(self):
        r'''
        Compute the downward and upward gaseous transmittances :math:`T_g^{\downarrow}` and
        :math:`T_g^{\uparrow}` (:ref:`Gaseous transmittance <methods>`) from the surface pressure and the
        water vapour, ozone and nitrogen dioxide columns of the input dataset.

        Sets the attributes ``Tg_d`` (air mass :math:`1/\mu_0`), ``Tg_u`` (air mass :math:`1/\mu_v`) and
        ``Tg_du`` (Sun-surface-sensor path, air mass :math:`1/\mu_0 + 1/\mu_v`). ``Tg_du`` equals
        ``Tg_d * Tg_u`` for the REPTRAN table; with the line-by-line O2 (``o2_model='lbl'``) it is the
        transmittance of the whole path averaged in the cells of the LUT, which is not the product of the
        one-way averages since the lines are narrower than the cells.

        Also sets the transmittances applied to each term of the signal: ``Tg_Ed`` (downwelling irradiance),
        ``Tg_atm`` (path reflectance ``Ratm``) and ``Tg_w`` (water-leaving term). They are ``Tg_d``,
        ``Tg_du`` and ``Tg_du`` with ``gas_coupling='decoupled'``, and the transmittances for the effective
        air masses of :py:class:`~radcalnet_oc.kernel.GasCoupling` with ``'effective_air_mass'``.
        '''
        input_db = self.input_db
        self.gas_trans.gas_tc['h2o'] = input_db.tcwv
        self.gas_trans.pressure = input_db['pressure']
        # gas_trans.gas_tc['h2o'] = tcwv
        self.gas_trans.gas_tc['o3'] = input_db['tco3']
        # gas_trans.gas_tc['ch4'] = tcch4
        self.gas_trans.gas_tc['no2'] = input_db['tcno2']

        self.gas_trans.air_mass = 1. / self.mu0
        self.Tg_d = self.gas_trans.get_gaseous_transmittance()

        self.gas_trans.air_mass = 1. / self.muv
        self.Tg_u = self.gas_trans.get_gaseous_transmittance()

        self.gas_trans.air_mass = 1. / self.mu0 + 1. / self.muv
        self.Tg_du = self.gas_trans.get_gaseous_transmittance()

        if self.coupling is None:
            self.Tg_Ed, self.Tg_atm, self.Tg_w = self.Tg_d, self.Tg_du, self.Tg_du
        else:
            aot = self.aot550
            m0, m = 1. / self.mu0, 1. / self.mu0 + 1. / self.muv
            gt, cp = self.gas_trans, self.coupling
            self.Tg_Ed = gt.get_coupled_transmittance(m0, 'Ed', aot, cp)
            self.Tg_atm = gt.get_coupled_transmittance(m, 'atm', aot, cp)
            self.Tg_w = gt.get_coupled_transmittance(m, 'w', aot, cp)

    def get_direct_transmittance(self,
                                 aot,
                                 rot,
                                 air_mass):
        r'''
        Direct (beam) transmittance including gaseous absorption, Rayleigh and aerosol extinction:
        :math:`T_g \exp\left[-(\tau_r + \tau_a)\, m\right]`.

        :param aot: aerosol optical thickness :math:`\tau_a`
        :param rot: Rayleigh optical thickness :math:`\tau_r`
        :param air_mass: air mass :math:`m` of the path (e.g. :math:`1/\mu_0 + 1/\mu_v`)
        :return: direct transmittance at the wavelengths of the gaseous LUT
        '''

        input_db = self.input_db
        self.gas_trans.gas_tc['h2o'] = input_db.tcwv
        self.gas_trans.pressure = input_db['pressure']
        # gas_trans.gas_tc['h2o'] = tcwv
        self.gas_trans.gas_tc['o3'] = input_db['tco3']
        # gas_trans.gas_tc['ch4'] = tcch4
        self.gas_trans.gas_tc['no2'] = input_db['tcno2']

        self.gas_trans.air_mass = air_mass
        Tg_dir = self.gas_trans.get_gaseous_transmittance()
        Tdir = np.exp(-(rot + aot) * air_mass)

        return Tg_dir * Tdir


    def get_irradiance_transmittance(self):
        r'''
        Interpolate the total (direct + diffuse) irradiance transmittance :math:`T^{\downarrow}` at the
        aerosol optical thickness at 550 nm of each measurement and at the wavelengths of the simulation.

        Requires the LUT to be prepared (:py:meth:`lut_preparation`). Sets the attribute ``Tra_d``.
        '''
        self.Tra_d = self.lut.trans_Ed.interp(  # sza=self.sza,
            aot_ref=self.aot550
        ).interpolate_na('time'
                         ).interp(wl=self.full_wl, method='quadratic')

    def get_radiance_transmittance(self):
        r'''
        Interpolate the upward radiance transmittance :math:`t^{\uparrow}` at the aerosol optical thickness
        at 550 nm of each measurement and at the wavelengths of the simulation.

        Requires the LUT to be prepared (:py:meth:`lut_preparation`). Sets the attribute ``tra_u``.
        '''

        tra_u = self.lut.trans_Lu.interp(aot_ref=self.aot550)#.interp(vza=self.vza)
        self.tra_u = tra_u.interp(wl=self.full_wl, method='quadratic')

    def get_downwelling_irradiance(self,
                                   aerosol_combination=AEROSOL_COMBINATION):
        r'''
        Compute the downwelling plane irradiance at the bottom of the atmosphere,
        :math:`E_d = T^{\downarrow}\, T_g^{\downarrow}\, \mu_0\, d^2\, F_0` (mW m-2 nm-1).

        Requires the LUT to be prepared (:py:meth:`lut_preparation`). Sets the attribute ``Ed``.

        :param aerosol_combination: not used (the aerosol models are those of the prepared LUT)
        '''

        self.get_gas_transmittance()

        # atmospheric aerosol-Rayleigh irradiance transmittance
        # self.lut.lut_preparation(sza=self.sza,
        #                         vza=[0],
        #                        aerosol_combination=aerosol_combination)
        self.get_irradiance_transmittance()

        self.Ed = self.Tra_d * self.Tg_Ed * self.mu0 * self.F0 * self.D2

    def lut_preparation(self, aerosol_combination=AEROSOL_COMBINATION):
        '''
        Prepare the look-up tables for the solar zenith angles of the input (rounded to 0.1 deg) and the
        viewing zenith angles, with the given proportions of the aerosol models.

        :param aerosol_combination: proportion of each aerosol model of ``config.yml``
            (``settings: aerosol_models``), list, array or :py:class:`xarray.DataArray`
        '''
        self.sza_lut = np.sort(np.unique(np.round(self.sza, 1)))
        self.lut.lut_preparation(wind=float(self.wind.mean()),
                                 sza=self.sza_lut,
                                 vza=self.vza.values,
                                 aerosol_combination=aerosol_combination)

    def execute(self,
                aerosol_combination=AEROSOL_COMBINATION,
                sunglint=None,
                sunglint_kw=None
                ):
        '''
        Compute the top-of-atmosphere simulation for the input time series (see :doc:`/methods`).

        The result is stored in ``radcalnet_db``, an :py:class:`xarray.Dataset` with the variables
        ``Rtoa`` (TOA reflectance), ``Ratm`` (atmospheric reflectance), ``Ed`` and ``E0`` (irradiance at the
        bottom and top of atmosphere, mW m-2 nm-1), ``Rrs`` (sr-1), ``D2``, ``aerosol_combination`` and the
        input atmospheric parameters.

        :param aerosol_combination: proportion of each aerosol model of ``config.yml``
            (``settings: aerosol_models``), list, array or :py:class:`xarray.DataArray`
            (e.g. retrieved with :py:class:`~radcalnet_oc.kernel.Aerosol`)
        :param sunglint: if set, wave slope statistics of the direct sunglint (e.g. ``'cm_iso_mix'``), computed
            with :py:meth:`compute_sunglint` (variables ``Rglint``, ``glint_angle``, ``glint_flags``...); ``Rtoa``
            does not include the direct glint
        :param sunglint_kw: keyword arguments of :py:meth:`compute_sunglint`
        '''

        t0 = time.perf_counter()
        attrs = self.provenance()
        logger.info('TOA simulation of %d measurement(s), %d viewing geometry(ies); TOA LUT %s, solar spectrum %s, '
                    'O2 %s, gas coupling %s, Rrs extension %s, Rayleigh pressure scaling %s',
                    self.sza.size, self.vza.size * np.size(self.azi), attrs['toa_lut'], self.solar_database,
                    attrs['o2_model'], attrs['gas_coupling'], attrs['rrs_extension'],
                    'on' if self.rayleigh_pressure else 'off')

        # atmospheric + sky-reflection radiance
        self.lut.lut_preparation(wind=self.wind,
                                 sza=self.sza,
                                 vza=self.vza.values,
                                 azi=self.azi,
                                 aerosol_combination=aerosol_combination)

        self.get_gas_transmittance()
        self.get_irradiance_transmittance()
        self.get_radiance_transmittance()

        # squeeze the single geometries but keep the time dimension, also for a single measurement
        Rdiff = self.lut.Rdiff_lut
        Rdiff = Rdiff.squeeze([d for d in Rdiff.dims if d != 'time' and Rdiff.sizes[d] == 1])
        Ratm = interp_wl_loglog(Rdiff.interp(aot_ref=self.aot550), self.full_wl)
        if self.rayleigh_pressure:
            dRray, fd, fu = self.rayleigh_pressure_correction()
            Ratm = Ratm + dRray
            self.Tra_d = self.Tra_d * fd
            self.tra_u = self.tra_u * fu
        # Sun-surface-sensor gaseous transmittance (Tg_d * Tg_u, computed for the whole path, or with the
        # effective air mass of the path reflectance, see get_gas_transmittance)
        Ratm = self.Tg_atm * Ratm
        self.Ratm = Ratm

        E0 = self.mu0 * self.F0 * self.D2
        self.E0 = E0
        Ed = self.Tra_d * self.Tg_Ed * E0
        # Ed = Ed.dropna('wl')

        # Rrs at the wavelengths of the simulation, extended beyond the range of the input
        Rrs, self.bbp_nir = extend_rrs(self.Rrs, self.full_wl.values, method=self.rrs_extension)
        Rrs = Rrs.assign_coords(wl=self.full_wl.values)
        self.Lw_boa = Rrs * Ed

        # = tra_u Tg_u Lw_boa, with the gaseous transmittance of the whole Sun-surface-sensor path (or with
        # the effective air mass of the water-leaving term)
        Lw_toa = self.tra_u * self.Tg_w * self.Tra_d * E0 * Rrs
        Lw_toa = Lw_toa.fillna(0)

        Rtoa = np.pi * Lw_toa / E0 + Ratm
        self.Rtoa = Rtoa

        Ed.name = 'Ed'

        radcalnet_db = Ed.to_dataset()  # .reset_coords('sza')
        self.radcalnet_db = radcalnet_db
        radcalnet_db['Ed'].attrs = {'unit': 'mW m-2 nm-1',
                                    'description': 'Plane solar irradiance at bottom-of-atmosphere'}

        radcalnet_db['Ratm'] = Ratm.reset_coords(drop=True)
        radcalnet_db['Ratm'].attrs = {'unit': '-',
                                      'description': 'Atmosphere (intrinsic + surface reflected) reflectance at top-of-atmosphere'}

        # radcalnet_db['mu0'] = self.mu0.reset_coords(drop=True)

        radcalnet_db['Rrs'] = Rrs.reset_coords(drop=True)
        radcalnet_db['Rrs'].attrs = {'unit': 'sr-1',
                                     'description': 'Remote sensing reflectance',
                                     'source': f'input Rrs interpolated linearly; beyond its range: '
                                               f'{self.rrs_extension} extension (radcalnet_oc.water.extend_rrs)'}
        if self.rrs_extension == 'pure_water':
            radcalnet_db['bbp_nir'] = self.bbp_nir.reset_coords(drop=True)

        radcalnet_db['E0'] = E0.reset_coords(drop=True)
        radcalnet_db['E0'].attrs = {'unit': 'mW m-2 nm-1',
                                    'description': 'Plane solar irradiance at top-of-atmosphere'}

        # radcalnet_db['Lw_toa'] = Lw_toa.reset_coords(drop=True)
        # radcalnet_db['Lw_toa'].attrs = {'unit': 'mW m-2 sr-1 nm-1',
        #                                'description': 'water-leaving radiance at top-of-atmosphere'}

        radcalnet_db['Rtoa'] = Rtoa.reset_coords(drop=True)
        radcalnet_db['Rtoa'].attrs = {'unit': '-',
                                      'description': 'Total reflectance at top-of-atmosphere'}

        radcalnet_db = xr.merge([radcalnet_db,
                                 self.input_db[['tcwv', 'tco3', 'tcno2', 'pressure',
                                                'aot550', 'aot865', 'ang_exp']]])
        radcalnet_db['wind'] = self.wind.reset_coords(drop=True)
        radcalnet_db['wind'].attrs = {'unit': 'm s-1', 'description': 'wind speed used for the sea-surface roughness'}

        radcalnet_db['aerosol_combination'] = self.lut.aerosol_combination
        radcalnet_db['aerosol_combination'].attrs = {'description':
                                                         'relative proportion of each aerosol model used for the computation',
                                                     'models': AEROSOL_MODELS}
        radcalnet_db['D2'] = self.D2

        # add sza as coordinates
        radcalnet_db['sza'] = self.sza
        radcalnet_db = radcalnet_db.set_coords('sza')

        radcalnet_db.attrs.update(attrs)
        self.radcalnet_db = radcalnet_db.squeeze()
        if sunglint is not None:
            self.compute_sunglint(stats=sunglint, **(sunglint_kw or {}))
        logger.info('TOA simulation done in %.1f s', time.perf_counter() - t0)

    def compute_sunglint(self, stats='cm_iso_mix', wl_ref=865., levels=None, angle_min=None, uncertainty=True,
                         threshold=None, wind_unc=None, ensemble=None, tg_min=0.9, **kwargs):
        r'''
        Direct sunglint at the TOA and sunglint flags, added to ``radcalnet_db`` (after :py:meth:`execute`).

        The direct sunlight reflected by the waves (:py:class:`~radcalnet_oc.coxmunk.Sunglint`, wind of each
        measurement) is attenuated by the direct transmittance of the Sun-surface-sensor path,
        :math:`T_g \exp[-(\tau_r + \tau_a)(1/\mu_0 + 1/\mu_v)]`, and is not included in the look-up tables
        (see :doc:`/methods`). Variables added: ``Rglint`` (TOA direct glint), ``Rg_surface`` (surface glint),
        ``Rtoa_glint`` (``Rtoa + Rglint``), ``glint_angle``, ``glint_fraction`` (``Rglint / Rtoa_glint`` at
        ``wl_ref``) and ``glint_flags`` (bitmask :data:`~radcalnet_oc.coxmunk.GLINT_FLAGS`: cumulative levels
        low, moderate and high of ``glint_fraction``, and near-specular geometry), e.g. to keep the low-glint
        dates: ``db.where(db.glint_flags & GLINT_FLAGS['glint_moderate'] == 0)``.

        With ``uncertainty``, the uncertainty of the glint is the half range of the glint of the plausible slope
        statistics (:data:`~radcalnet_oc.coxmunk.GLINT_ENSEMBLE`) at the wind and at the wind :math:`\pm`
        ``wind_unc``, attenuated as the glint (``Rglint_unc``, spectral), and the spectral flag ``glint_uncertain``
        (0/1 per wavelength) marks the wavelengths where ``Rglint_unc / Rtoa_glint`` exceeds ``threshold`` (5 %):
        bands of the simulation too uncertain for Cal/Val, whatever the glint level, e.g. the SWIR of a date whose
        visible bands are usable. ``glint_unc_rel`` and ``glint_unc_wl`` give the largest ratio in the atmospheric
        windows (gaseous transmittance of the path > ``tg_min``) and its wavelength.

        :param stats: wave slope statistics (``'cm_iso'``, ``'cm_iso_t'``, ``'cm_iso_mix'``, ``'cm_dir'``,
            ``'bh2006'``); the default heavy-tailed mixture fits the Sentinel-2 matchups of the AERONET-OC
            platforms better than ``'cm_iso'``
        :param wl_ref: reference wavelength of the glint fraction (nm)
        :param levels: glint fractions of the levels low, moderate and high
            (default :data:`~radcalnet_oc.coxmunk.GLINT_LEVELS`)
        :param angle_min: glint angle below which the near-specular bit is set
            (default :data:`~radcalnet_oc.coxmunk.GLINT_ANGLE_MIN`)
        :param uncertainty: compute the glint uncertainty and the ``glint_uncertain`` bit
        :param threshold: relative uncertainty of the TOA reflectance above which ``glint_uncertain`` is set
            (default :data:`~radcalnet_oc.coxmunk.GLINT_UNC_THRESHOLD`, 0.05)
        :param wind_unc: uncertainty of the wind speed (m s-1, default :data:`~radcalnet_oc.coxmunk.WIND_UNC`)
        :param ensemble: plausible slope statistics, sequence of ``(stats, kwargs)``
            (default :data:`~radcalnet_oc.coxmunk.GLINT_ENSEMBLE`)
        :param tg_min: minimum gaseous transmittance of the path of the wavelengths of ``glint_unc_rel``
        :param kwargs: parameters of the statistics passed to :py:meth:`~radcalnet_oc.coxmunk.Sunglint.sunglint`
            (``nu``, ``f_tail``, ``sigma2_tail``, ``sigma2_coef``, ``wazi``, ``shadow``)
        '''
        levels = coxmunk.GLINT_LEVELS if levels is None else levels
        angle_min = coxmunk.GLINT_ANGLE_MIN if angle_min is None else angle_min
        threshold = coxmunk.GLINT_UNC_THRESHOLD if threshold is None else threshold
        wind_unc = coxmunk.WIND_UNC if wind_unc is None else wind_unc
        ensemble = coxmunk.GLINT_ENSEMBLE if ensemble is None else ensemble
        sza, wind = self.sza.values, self.wind.values
        vza, azi = np.atleast_1d(self.vza.values), np.atleast_1d(self.azi.values)
        Rg = np.zeros((len(sza), len(vza), len(azi)))
        Rg_unc = np.zeros_like(Rg)
        for i, (s, w) in enumerate(zip(sza, wind)):
            for j, v in enumerate(vza):
                for k, a in enumerate(azi):
                    # Sunglint: relative azimuth 180 deg in the specular direction (180 - azi of the LUT)
                    g = coxmunk.Sunglint(float(s), float(v), 180. - float(a))
                    Rg[i, j, k] = max(g.sunglint(float(w), stats=stats, **kwargs)[0], 0.)
                    if uncertainty:
                        Rg_unc[i, j, k] = coxmunk.glint_uncertainty(float(s), float(v), 180. - float(a), float(w),
                                                                    ensemble, wind_unc)[0]
        coords = {'time': self.sza.time.values, 'vza': vza, 'azi': azi}
        Rg = xr.DataArray(Rg, coords=coords, dims=('time', 'vza', 'azi'))
        Rg_unc = xr.DataArray(Rg_unc, coords=coords, dims=('time', 'vza', 'azi'))
        angle = xr.DataArray(coxmunk.glint_angle(sza[:, None, None], vza[None, :, None], azi[None, None, :]),
                             coords=coords, dims=('time', 'vza', 'azi'))

        full_wl = self.full_wl
        muv = xr.DataArray(np.cos(np.radians(vza)), coords={'vza': vza}, dims='vza')
        tau_r = self.lut.rot.interp(wl=full_wl, kwargs={'fill_value': 'extrapolate'}) * \
            self.input_db['pressure'] / 1013.25
        tau_a = self.lut.aot_lut.interp(aot_ref=self.aot550).interp(wl=full_wl, method='quadratic')
        Tdir = np.exp(-(tau_r + tau_a) * (1 / self.mu0 + 1 / muv))
        Tg = (self.Tg_d * self.Tg_u).sel(wl=full_wl)
        Rglint = (Tg * Tdir * Rg).reset_coords(drop=True)
        Rglint = Rglint.transpose(*[d for d in self.Rtoa.dims if d in Rglint.dims])
        total = self.Rtoa.reset_coords(drop=True) + Rglint
        fraction = (Rglint / total).sel(wl=wl_ref, method='nearest', drop=True)
        if uncertainty:
            Rglint_unc = (Tg * Tdir * Rg_unc).reset_coords(drop=True).transpose(*Rglint.dims)
            ratio = Rglint_unc / total
            uncertain = (ratio > threshold).astype(np.uint8)
            rel = ratio.where((Tg > tg_min).reset_coords(drop=True))
            unc_rel = rel.max('wl')
            unc_wl = rel.fillna(-1).idxmax('wl')
        flags = xr.DataArray(coxmunk.glint_flags(fraction.values, angle.broadcast_like(fraction).values,
                                                 levels, angle_min),
                             coords=fraction.coords, dims=fraction.dims)

        db = self.radcalnet_db
        params = {'stats': stats, **kwargs}
        db['Rglint'] = Rglint.squeeze()
        db['Rglint'].attrs = {'unit': '-', 'description': 'direct sunglint reflectance at top-of-atmosphere '
                              '(not included in Rtoa)', 'glint_statistics': repr(params)}
        db['Rg_surface'] = Rg.squeeze()
        db['Rg_surface'].attrs = {'unit': '-', 'description': 'direct sunglint reflectance at the surface',
                                  'glint_statistics': repr(params)}
        db['Rtoa_glint'] = total.squeeze()
        db['Rtoa_glint'].attrs = {'unit': '-', 'description': 'Total reflectance at top-of-atmosphere with the '
                                  'direct sunglint'}
        db['glint_angle'] = angle.squeeze()
        db['glint_angle'].attrs = {'unit': 'deg', 'description': 'angle between the viewing direction and the '
                                   'specular reflection of the Sun'}
        db['glint_fraction'] = fraction.squeeze()
        db['glint_fraction'].attrs = {'unit': '-', 'description': f'Rglint / Rtoa_glint at {wl_ref:g} nm'}
        if uncertainty:
            db['Rglint_unc'] = Rglint_unc.squeeze()
            db['Rglint_unc'].attrs = {'unit': '-', 'description': 'uncertainty of the TOA direct glint: half range of '
                                      f'the slope statistics {[s for s, _ in ensemble]} at the wind +/- {wind_unc:g} '
                                      'm s-1', 'ensemble': repr(ensemble)}
            db['glint_uncertain'] = uncertain.squeeze()
            db['glint_uncertain'].attrs = coxmunk.uncertain_attrs(threshold, wind_unc, ensemble)
            db['glint_unc_rel'] = unc_rel.transpose(*fraction.dims).squeeze()
            db['glint_unc_rel'].attrs = {'unit': '-', 'description': 'largest Rglint_unc / Rtoa_glint in the '
                                         f'atmospheric windows (path gaseous transmittance > {tg_min:g})'}
            db['glint_unc_wl'] = unc_wl.transpose(*fraction.dims).squeeze()
            db['glint_unc_wl'].attrs = {'unit': 'nm', 'description': 'wavelength of glint_unc_rel'}
        db['glint_flags'] = flags.squeeze()
        db['glint_flags'].attrs = coxmunk.flag_attrs(levels, angle_min, wl_ref)
        self.radcalnet_db = db
        return db

    def rayleigh_pressure_correction(self):
        r"""
        Change of the Rayleigh scattering from the pressure of the look-up tables (:math:`P_0` = 1015.2 hPa, ``LUT_PRESSURE``) to
        the surface pressure :math:`P` of each measurement (after :py:meth:`~radcalnet_oc.lut.LUT.lut_preparation`).

        The Rayleigh optical thickness scales as :math:`\tau_r(P) = \tau_r(P_0)\,P/P_0`. The Rayleigh reflectance of
        the tables (molecules and sea surface, no aerosol) is a smooth function of :math:`\tau_r` at a given
        geometry; its logarithmic sensitivity :math:`D = d\ln R_{ray}/d\ln\tau_r` (1 in single scattering, less
        when multiple scattering matters) is computed from the wavelength nodes of the tables, where
        :math:`\tau_r` varies, and

        .. math::

           \Delta R_{ray} = R_{ray}\left[(P/P_0)^D - 1\right]

        is added to the path reflectance (the coupling between the aerosols and the molecules is kept, first
        order). The diffuse transmittances of the irradiance and of the water-leaving radiance are multiplied by
        :math:`\exp(-\Delta\tau_r/2\mu)`, half of the Rayleigh scattering being lost for the direct beam.

        :return: :math:`\Delta R_{ray}` (wavelengths of the simulation), factors of ``Tra_d`` and ``tra_u``
        """
        Rray = self.lut.Rray
        Rray = Rray.squeeze([d for d in Rray.dims if d != 'time' and Rray.sizes[d] == 1])
        tau = self.lut.rot.interp(wl=Rray.wl, kwargs={'fill_value': 'extrapolate'})
        # logarithmic derivative along the wavelength nodes (tau_r and R_ray decrease with the wavelength)
        D = xr.apply_ufunc(lambda r: np.gradient(np.log(np.clip(r, 1e-12, None)), np.log(tau.values), axis=-1),
                           Rray, input_core_dims=[['wl']], output_core_dims=[['wl']])
        x = self.input_db['pressure'] / LUT_PRESSURE
        full_wl = self.full_wl
        Rray_full = interp_wl_loglog(Rray, full_wl)
        D_full = D.interp(wl=full_wl)
        dRray = (Rray_full * (x ** D_full - 1)).reset_coords(drop=True)
        dtau = self.lut.rot.interp(wl=full_wl, kwargs={'fill_value': 'extrapolate'}) * (x - 1)
        fd = np.exp(-dtau / (2 * self.mu0)).reset_coords(drop=True)
        fu = np.exp(-dtau / (2 * self.muv)).reset_coords(drop=True)
        fu = fu.squeeze([d for d in fu.dims if d not in self.tra_u.dims and fu.sizes[d] == 1])
        self.dRray = dRray
        return dRray, fd, fu

    def provenance(self):
        '''
        Settings of the simulation, written as global attributes of ``radcalnet_db``: version of the package
        (:py:func:`package_version`), O2 model, gas coupling and its coefficient table, spectral extension
        of Rrs, solar spectrum, default wind speed and TOA look-up table (file and DOI of its Zenodo record,
        or the light LUT of the package).

        :return: dict of text and number attributes (NetCDF compatible)
        '''
        lut_file = Path(str(self.lut.lut_file))
        if lut_file == Path(str(LUT_FILE_BACKUP)):
            lut_doi = 'none (light LUT of the package)'
        else:
            record = (config.get('zenodo') or {}).get('record')
            lut_doi = f'10.5281/zenodo.{record}' if record else 'unknown'
        return {'radcalnet_oc_version': package_version(),
                'o2_model': self.gas_trans.o2_model,
                'gas_coupling': self.gas_coupling,
                'gas_coupling_table': Path(str(self.gas_coupling_file)).name if self.gas_coupling_file else 'none',
                'rrs_extension': self.rrs_extension,
                'rayleigh_pressure': int(self.rayleigh_pressure),
                'solar_database': self.solar_database,
                'default_wind': float(WIND),
                'toa_lut': lut_file.name,
                'toa_lut_doi': lut_doi}
