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

        self.input_db = input_db
        self.vza = vza
        self.azi = azi

        # get LUT object
        lut = LUT()

        # get full spectral resolution wavelength from gas LUT
        # crop to 350 - 2500 range to comply with OSOAA lut
        full_wl = lut.gas_lut.wl.sel(wl=slice(350, 2500))
        lut.wl=full_wl

        lut.load_auxiliary_data()
        solar_irr = SolarIrradiance()

        self.F0 = solar_irr.__dict__[solar_database].interp(wl=full_wl)
        self.gas_trans = GaseousTransmittance(lut.gas_lut)

        self.lut_Ttot_Ed = lut.trans_lut.Ttot_Ed.isel(wind=0)

    def set_param(self):

        input_db = self.input_db
        self.sza = input_db.sza
        self.mu0 = np.cos(np.radians(self.sza))
        self.muv = np.cos(np.radians(self.vza))

        self.Lw_boa = input_db['Lw']
        self.aot550 = input_db['aot550']

        # get correction for Sun-Earth distance
        self.D2 = Misc.earth_sun_correction(input_db['day_of_year'])

    def get_gas_transmittance(self):

        input_db = self.input_db
        self.gas_trans.gas_tc['h2o'] = input_db.tcwv
        self.gas_trans.pressure = input_db['pressure']
        # gas_trans.gas_tc['h2o'] = tcwv
        self.gas_trans.gas_tc['o3'] = input_db['tco3']
        # gas_trans.gas_tc['ch4'] = tcch4
        self.gas_trans.gas_tc['no2'] = input_db['tcno2']

        self. gas_trans.air_mass = 1. / self.mu0
        self.Tg_d = self.gas_trans.get_gaseous_transmittance()

        self.gas_trans.air_mass = 1. / self.muv
        self.Tg_u = self.gas_trans.get_gaseous_transmittance()
