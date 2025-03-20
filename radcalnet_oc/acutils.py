'''
Atmospheric Correction utilities to manage LUT and atmosphere parameters (aerosols, gases)
'''

import os, sys
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


class Aerosol:
    '''
    aerosol parameters and parameterizations
    '''

    def __init__(self):
        self.aot550 = 0.1
        self.wavelengths = []
        self.aot = []
        self.wl = []
        self.ang = []
        self.o3du = 300
        self.no2du = 300
        self.fcoef = 0.5
        self.popt = []

    def func(self, lnwl, a, b, c):
        '''function for spectral variation of AOT'''

        return (a + b * lnwl + c * lnwl ** 2)

    def fit_spectral_aot(self, wl, aot):
        '''call to get fitting results on AOT data'''
        lnwl = np.log(wl)
        self.popt, pcov = curve_fit(self.func, lnwl, np.log(aot))

    def get_spectral_aot(self, wl):
        '''set aot for a given set of wavelengths'''
        lnwl = np.log(wl)
        return np.exp(self.func(lnwl, *self.popt))

    def func_aero(self, Cext, fcoef):
        '''function to fit spectral behavior of bimodal aerosols
         onto aeronet optical thickness'''
        return fcoef * Cext[0] + (1 - fcoef) * Cext[1]

    def fit_aero(self, nCext_f, nCext_c, naot):
        '''Call to get fine mode coefficient based on fitting on AOT data.

        Arguments:
          * ``nCext_f`` -- Normalized extinction coefficient of the fine mode aerosols
          * ``nCext_c`` -- Normalized extinction coefficient of the coarse mode aerosols
          * ``naot``    -- Normalized spectral aerosol optical thickness

        Return values:
          The mixing ratio of the fine mode aerosol

        Notes:
              .
            '''
        self.fcoef, pcov = curve_fit(self.func_aero, [nCext_f, nCext_c], naot)
        return self.fcoef


class CamsParams:
    def __init__(self,
                 name,
                 resol):
        self.name = name
        self.resol = resol


class Gases():
    '''
     Intermediate class to set parameters for absorbing gases.
    '''

    def __init__(self):
        self.pressure = 1010
        self.pressure_gas_ref = 1000

        self.gas_tc = {'co2': 1,
                       'o2': 1,
                       'o4': 1,
                       'ch4': 1e-2,
                       'no2': 3e-6,
                       'o3': 6.5e-3,
                       'h2o': 30}
        self.coef_abs_scat = {'co2': 0.4,
                              'o2': 0.3,
                              'o4': 0.3,
                              'ch4': 0.5,
                              'no2': 1,
                              'o3': 1,
                              'h2o': 0.3}


class GaseousTransmittance(Gases):

    def __init__(self,
                 gas_lut: xr.DataArray,
                 zenith_angle=0
                 ):
        '''
        Class containing functions to compute the direct transmittance of the absorbing gases.
        :param gas_lut: xarray.DataArray Look-up table data for gaseous absorption
        :param zenith_angle: zenith angle (solar or viewing) in degrees
        '''
        Gases.__init__(self)
        self.air_mass = np.cos(np.radians(zenith_angle))
        self.gas_lut = gas_lut

    def Tgas_background(self):
        '''
        Compute direct transmittance for background absorbing gases: :math:`CO, O_2, O_4`

        :return:
        '''
        gl = self.gas_lut
        self.ot_air = self.pressure / self.pressure_gas_ref * \
                      (gl.co + self.coef_abs_scat['co2'] * gl.co2 +
                       self.coef_abs_scat['o2'] * gl.o2 +
                       self.coef_abs_scat['o4'] * gl.o4)
        self.Tg_bg = np.exp(- self.air_mass * self.ot_air)
        return self.Tg_bg

    def Tgas(self,
             gas_name,
             ):
        '''
        Compute hyperspectral transmittance for a given absorbing gas and
        convolve it with the spectral response functions of the satellite sensor.

        :param gas_name: name of the absorbing gas, choose between:
            - 'h2o'
            - 'o3'
            - 'n2o'
        :return: Gaseous transmittance for satellite bands
        '''

        ot = self.coef_abs_scat[gas_name] * self.gas_tc[gas_name] * self.gas_lut[gas_name]
        Tg = np.exp(- self.air_mass * ot)
        return Tg

    def get_gaseous_transmittance(self,
                                  gases=['ch4', 'no2', 'o3', 'h2o'],
                                  background=True):
        '''
        Get the final total gaseous transmittance.
        :return:
        '''

        first = True
        for gas_name in gases:
            if first:
                Tg_tot = self.Tgas(gas_name)
                first = False
            else:
                Tg_tot = Tg_tot * self.Tgas(gas_name)

        if background:
            Tg_tot = Tg_tot * self.Tgas_background()

        return Tg_tot


class Misc:
    '''
    Miscelaneous utilities
    '''

    @staticmethod
    def get_pressure(alt, psl):
        '''Compute the pressure for a given altitude
           alt : altitude in meters (float or np.array)
           psl : pressure at sea level in hPa
           palt : pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))
