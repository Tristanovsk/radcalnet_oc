'''
Simulation of the top-of-atmosphere signal from surface measurements.

:py:class:`Process` chains the steps described in :doc:`/methods`: solar irradiance, gaseous and
atmospheric transmittances, downwelling irradiance, water-leaving radiance, atmospheric path
reflectance and top-of-atmosphere reflectance.
'''

import os

import numpy as np
import xarray as xr

import logging

from importlib.resources import files
import yaml

from . import GaseousTransmittance, Misc, LUT, SolarIrradiance

template = files('radcalnet_oc.data.template').joinpath('template_clear_water.nc')
# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']


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
                 Rrs_name='Rrs'
                 ):
        '''
        :param input_db: time series of the surface measurements, :py:class:`xarray.Dataset` with a
            ``time`` dimension and the variables listed in :doc:`/usage` (``Rrs``, ``sza``,
            ``day_of_year``, ``aot550``, ``pressure``, ``tcwv``, ``tco3``, ``tcno2``...);
            if None, the clear-water template of the package is used
        :param vza: viewing zenith angles (deg), :py:class:`xarray.DataArray` with a ``vza`` dimension
        :param azi: relative azimuth angles (deg), :py:class:`xarray.DataArray` with an ``azi`` dimension
        :param central_wl: not used (the wavelengths are those of the gaseous absorption LUT)
        :param solar_database: solar spectrum used for :math:`F_0`: ``'tsis'`` (default), ``'thuillier'``,
            ``'gueymard'`` or ``'kurucz'``
        :param Rrs_name: name of the remote-sensing reflectance variable in ``input_db``
        '''

        if input_db is None:
            input_db = xr.open_dataset(template)

        self.input_db = input_db

        self.vza = vza
        self.azi = azi

        # get LUT object
        logging.info('get LUT object')
        lut = LUT()
        self.lut = lut

        # get full spectral resolution wavelength from gas LUT
        # crop to 350 - 2500 range to comply with OSOAA lut
        full_wl = lut.gas_lut.wl.sel(wl=slice(350, 2500))
        self.full_wl = full_wl
        # lut.wl = full_wl

        logging.info('get auxiliary data')
        lut.load_auxiliary_data()

        logging.info('get solar irradiance')
        solar_irr = SolarIrradiance()
        self.F0 = solar_irr.__dict__[solar_database].interp(wl=full_wl)

        logging.info('get gaseous transmittance object')
        self.gas_trans = GaseousTransmittance(lut.gas_lut)

        logging.info('set input parameters')
        self.set_param(Rrs_name=Rrs_name)

    def set_param(self,
                  Rrs_name='Rrs'):
        '''
        Read the solar geometry, the remote-sensing reflectance, the aerosol optical thickness and the
        Earth-Sun distance correction from the input dataset.

        :param Rrs_name: name of the remote-sensing reflectance variable in ``input_db``

        Sets the attributes ``sza``, ``mu0``, ``muv``, ``Rrs``, ``aot550`` and ``D2``.
        '''

        input_db = self.input_db
        self.sza = input_db.sza
        self.mu0 = np.cos(np.radians(self.sza))
        self.muv = np.cos(np.radians(self.vza))

        self.Rrs = input_db[Rrs_name]
        # self.Rrs = self.Rrs.rename({"hyper_wl":"wl"})
        self.aot550 = input_db['aot550']

        # get correction for Sun-Earth distance
        self.D2 = Misc.earth_sun_correction(input_db['day_of_year'])

    def get_gas_transmittance(self):
        r'''
        Compute the downward and upward gaseous transmittances :math:`T_g^{\downarrow}` and
        :math:`T_g^{\uparrow}` (:ref:`Gaseous transmittance <methods>`) from the surface pressure and the
        water vapour, ozone and nitrogen dioxide columns of the input dataset.

        Sets the attributes ``Tg_d`` (air mass :math:`1/\mu_0`) and ``Tg_u`` (air mass :math:`1/\mu_v`).
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

        self.Ed = self.Tra_d * self.Tg_d * self.mu0 * self.F0 * self.D2

    def lut_preparation(self, aerosol_combination=AEROSOL_COMBINATION):
        '''
        Prepare the look-up tables for the solar zenith angles of the input (rounded to 0.1 deg) and the
        viewing zenith angles, with the given proportions of the aerosol models.

        :param aerosol_combination: proportion of each aerosol model of ``config.yml``
            (``settings: aerosol_models``), list, array or :py:class:`xarray.DataArray`
        '''
        self.sza_lut = np.sort(np.unique(np.round(self.sza, 1)))
        self.lut.lut_preparation(sza=self.sza_lut,
                                 vza=self.vza.values,
                                 aerosol_combination=aerosol_combination)

    def execute(self,
                aerosol_combination=AEROSOL_COMBINATION
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
        '''

        # atmospheric + sky-reflection radiance
        self.lut.lut_preparation(sza=self.sza,
                                 vza=self.vza.values,
                                 azi=self.azi,
                                 aerosol_combination=aerosol_combination)

        self.get_gas_transmittance()
        self.get_irradiance_transmittance()
        self.get_radiance_transmittance()

        Ratm = self.lut.Rdiff_lut.squeeze().interp(aot_ref=self.aot550
                                                   ).interp(wl=self.full_wl, method='quadratic')
        Ratm = self.Tg_d * self.Tg_u * Ratm
        self.Ratm = Ratm

        E0 = self.mu0 * self.F0 * self.D2
        self.E0 = E0
        Ed = self.Tra_d * self.Tg_d * E0
        # Ed = Ed.dropna('wl')

        Rrs = self.Rrs.interp(wl=self.full_wl).fillna(0)
        self.Lw_boa = Rrs * Ed

        Lw_toa = self.tra_u * self.Tg_u * self.Lw_boa
        Lw_toa = Lw_toa.fillna(0)

        Rtoa = np.pi * Lw_toa / E0 + Ratm

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
        radcalnet_db['Rrs'].attrs = {'unit': '-',
                                     'description': 'Remote sensing reflectance',
                                     'source': 'from AERONET-OC and spectral interpolation based on InvRrs'}

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

        radcalnet_db['aerosol_combination'] = self.lut.aerosol_combination
        radcalnet_db['aerosol_combination'].attrs = {'description':
                                                         'relative proportion of each aerosol model used for the computation',
                                                     'models': AEROSOL_MODELS}
        radcalnet_db['D2'] = self.D2

        # add sza as coordinates
        radcalnet_db['sza'] = self.sza
        radcalnet_db = radcalnet_db.set_coords('sza')

        self.radcalnet_db = radcalnet_db.squeeze()
