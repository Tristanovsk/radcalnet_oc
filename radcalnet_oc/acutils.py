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
        # atmosphere auxiliary data
        # TODO get them from CAMS
        self.pressure = 1010
        self.to3c = 6.5e-3
        self.tno2c = 3e-6
        self.tch4c = 1e-2
        self.psl = 1013
        self.coef_abs_scat = {'co2':0.4,
                              'o2':0.3,
                              'o4':0.3,
                              'ch4': 0.5,
                              'no2': 1,
                              'o3': 1,
                              'h2o':0.3}


class GaseousTransmittance(Gases):
    '''
    Class containing functions to compute rasters of the direct transmittance of the absorbing gases.
    '''

    def __init__(self,
                 gas_lut,
                 air_mass,
                 pressure,
                 ):

        Gases.__init__(self)

        self.gas_lut = gas_lut     
        
        self.air_mass = air_mass
        self.pressure = pressure
        #self.coef_abs_scat = 0.3
        self.Tg_tot_coarse = None
        self.cams_gases = {'ch4': CamsParams('tc_ch4', 4),
                           'no2': CamsParams('tcno2', 7),
                           'o3': CamsParams('gtco3', 4),
                           'h2o': CamsParams('tcwv', 1), }

    def Tgas_background(self):
        '''
        Compute direct transmittance for background absorbing gases: :math:`CO, O_2, O_4`

        :return:
        '''
        gl = self.gas_lut
        pressure = self.pressure.round(1)
        self.ot_air = (gl.co + self.coef_abs_scat['co2'] * gl.co2 +
                       self.coef_abs_scat['o2'] * gl.o2 +
                       self.coef_abs_scat['o4'] * gl.o4) / 1000

        wl_ref = gl.wl
        SRF_hr = self.prod.raster.SRF.interp(wl_hr=wl_ref.values)
        vals = np.unique(pressure)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 1:
            vals = np.concatenate([vals, 1.2 * vals])
        Tg_raster = []
        for val in vals:
            Tg = np.exp(- self.air_mass * self.ot_air * val)
            Tg = Tg.rename({'wl': 'wl_hr'})

            Tg_int = []
            for label, srf in SRF_hr.groupby('wl', squeeze=False):
                srf = srf.dropna('wl_hr').squeeze()
                Tg_ = Tg.sel(wl_hr=srf.wl_hr)
                wl_integr = Tg_.wl_hr.values

                Tg_ = np.trapz(Tg_ * srf, wl_integr) / np.trapz(srf, wl_integr)
                Tg_int.append(Tg_)
            Tg_raster.append(xr.DataArray(Tg_int, name='Ttot', coords={'wl': SRF_hr.wl.values}
                                          ).assign_coords({'pressure': val}))
        Tg_raster = xr.concat(Tg_raster, dim='pressure')
        return Tg_raster.interp(pressure=pressure).drop_vars(['pressure'])

    def Tgas(self,
             gas_name,
             coef_abs_scat=1):
        '''
        Compute hyperspectral transmittance for a given absorbing gas and
        convolve it with the spectral response functions of the satellite sensor.

        :param gas_name: name of the absorbing gas, choose between:
            - 'h2o'
            - 'o3'
            - 'n2o'
        :return: Gaseous transmittance for satellite bands
        '''


        cams_gas = self.cams_gases[gas_name].name
        resol = self.cams_gases[gas_name].resol
        lut_abs = self.gas_lut[gas_name]

        # round number to speed up computation
        # and scale the concentration following 6S approach due to vertical distribution
        # with scattering layer above absorbing gases
        rounded = coef_abs_scat * self.cams.raster[cams_gas].round(resol)

        wl_ref = self.gas_lut.wl
        SRF_hr = self.prod.raster.SRF.interp(wl_hr=wl_ref.values)
        vals = np.unique(rounded)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 1:
            vals = np.concatenate([vals, 1.2 * vals])
        Tg_raster = []
        for val in vals:
            Tg = np.exp(- self.air_mass * lut_abs * val)
            Tg = Tg.rename({'wl': 'wl_hr'})

            Tg_int = []
            for label, srf in SRF_hr.groupby('wl', squeeze=False):
                srf = srf.dropna('wl_hr').squeeze()
                Tg_ = Tg.sel(wl_hr=srf.wl_hr)
                wl_integr = Tg_.wl_hr.values

                Tg_ = np.trapz(Tg_ * srf, wl_integr) / np.trapz(srf, wl_integr)
                Tg_int.append(Tg_)
            Tg_raster.append(xr.DataArray(Tg_int, name='Ttot', coords={'wl': SRF_hr.wl.values}
                                          ).assign_coords({'tc': val}))
        Tg_raster = xr.concat(Tg_raster, dim='tc')
        return Tg_raster.interp(tc=rounded)

    def get_gaseous_optical_thickness(self):
        '''
        Get gaseous optival thickness from total column integrated concentration.
        :return:
        '''

        gas_lut = self.gas_lut

        ot_o3 = gas_lut.o3 * self.to3c
        ot_ch4 = gas_lut.ch4 * self.tch4c
        ot_no2 = gas_lut.no2 * self.tno2c
        ot_air = (gas_lut.co + self.coef_abs_scat['co2'] * gas_lut.co2 +
                  self.coef_abs_scat['o2'] * gas_lut.o2 +
                  self.coef_abs_scat['o4'] * gas_lut.o4) * self.pressure / 1000
        self.abs_gas_opt_thick = ot_ch4 + ot_no2 + ot_o3 + ot_air

    def get_gaseous_transmittance(self,
                                  gases=['ch4','no2','o3','h2o'],
                                  background=True):
        '''
        Get the final total gaseous transmittance.
        :return:
        '''

        first = True
        for gas in gases:
            if first:
                Tg_tot = self.Tgas(gas, self.coef_abs_scat[gas])
                first=False
            else:
                Tg_tot =Tg_tot *self.Tgas(gas,
                                      self.coef_abs_scat[gas])

        if background:
            Tg_tot = Tg_tot * self.Tgas_background()
        # Tg_other = Tg_other.rename({'longitude': 'x', 'latitude': 'y'})
        # Nx = len(Tg_other.x)
        # Ny = len(Tg_other.y)
        # x = np.linspace(self.xmin, self.xmax, Nx)
        # y = np.linspace(self.ymax, self.ymin, Ny)
        # Tg_other['x'] = x
        # Tg_other['y'] = y
        self.Tg_tot_coarse = Tg_tot
        # TODO remove interp for the whole object and proceed with loop on spectral bands to save memory
        return Tg_tot  # .interp(x=self.prod.raster.x, y=self.prod.raster.y)






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
