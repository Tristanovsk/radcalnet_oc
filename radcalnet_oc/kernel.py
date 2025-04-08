import os

import importlib_resources
import yaml

import numpy as np
import xarray as xr

import logging
import gc

from numba import njit

from . import AuxData, GaseousTransmittance, Misc, LUT, SolarIrradiance


class Kernel():
    def __init__(self,
                 input_db,
                 vza=0,
                 azi=0,
                 central_wl=np.arange(350, 2500, 1),
                 solar_database='tsis'
                 ):
        '''

        :param input_db:
        :param vza:
        :param azi:
        :param central_wl:
        :param solar_database:
        '''
        self.input_db = input_db
        self.vza = vza
        self.azi = azi

        # get LUT object
        lut = LUT()
        self.lut = lut

        # get full spectral resolution wavelength from gas LUT
        # crop to 350 - 2500 range to comply with OSOAA lut
        full_wl = lut.gas_lut.wl.sel(wl=slice(350, 2500))
        self.full_wl = full_wl
        lut.wl = full_wl

        lut.load_auxiliary_data()
        solar_irr = SolarIrradiance()

        self.F0 = solar_irr.__dict__[solar_database].interp(wl=full_wl)
        self.gas_trans = GaseousTransmittance(lut.gas_lut)

        self.set_param()

    def set_param(self):
        '''

        :return:
        '''
        input_db = self.input_db
        self.sza = input_db.sza
        self.mu0 = np.cos(np.radians(self.sza))
        self.muv = np.cos(np.radians(self.vza))

        self.Lw_boa = input_db['Lw']
        self.aot550 = input_db['aot550']

        # get correction for Sun-Earth distance
        self.D2 = Misc.earth_sun_correction(input_db['day_of_year'])

    def get_gas_transmittance(self):
        '''

        :return:
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

    def get_irradiance_transmittance(self):
        '''

        :return:
        '''
        self.Tra_d = self.lut.trans_aero_lut.interp(sza=self.sza,
                                             aot_ref=self.aot550
                                             )

    def get_radiance_transmittance(self):
        '''

        :return:
        '''

        tra_u = (self.lut.trans_aero_lut.interp(sza=self.vza, aot_ref=self.aot550) ** 1.07)
        tra_u = tra_u.interp(wl=self.full_wl, method='quadratic')
        self.tra_u = tra_u.rename({'sza': 'vza'})

    def get_downwelling_irradiance(self,
                                   aerosol_combination=[0, 0.5, 0.5, 0., 0.]):
        '''
        Function to get the downwelling irradiance at the bottom-of-atmosphere level.

        :return:
        '''


        self.get_gas_transmittance()

        # atmospheric aerosol-Rayleigh irradiance transmittance
        self.lut.lut_preparation(sza=self.sza,
                                 vza=[0],
                                 weights=aerosol_combination)
        self.get_irradiance_transmittance()

        self.Ed = self.Tra_d * self.Tg_d * self.mu0 * self.F0 * self.D2

    def execute(self,
                aerosol_combination=[0, 0.5, 0., 0., 0.]
                ):
        '''

        :return:
        '''


        self.get_gas_transmittance()
        self.get_irradiance_transmittance()
        self.get_radiance_transmittance()

        # atmospheric + sky-reflection radiance
        self.lut.lut_preparation(sza=self.sza, vza=self.vza, weights=aerosol_combination)
        Ratm = self.lut.Rdiff_lut.squeeze().interp(aot_ref=self.aot550)
        Ratm = self.Tg_d * self.Tg_u * Ratm

        Ed = Tra_d * Tg_d * mu0 * F0 * D2
        Ed = Ed.dropna('wl')

        Lw_toa = tra_u * Tg_u * Lw_boa.interp(wl=full_wl)
        Lw_toa = Lw_toa.fillna(0)

        Ed.name = 'Ed'
        radcalnet_db = Ed.reset_coords('sza')
        radcalnet_db['aot550'] = aot550
        radcalnet_db['D2'] = D2
        radcalnet_db['Ratm'] = Ratm.reset_coords(drop=True)
        radcalnet_db['mu0'] = mu0.reset_coords(drop=True)
        radcalnet_db['F0'] = F0.reset_coords(drop=True)
        radcalnet_db['Lw_toa'] = Lw_toa.reset_coords(drop=True)
