'''
Module to load LUT files.
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

# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / '../config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

GRSDATA = config['path']['grsdata']
TOALUT = config['path']['toa_lut']
TRANSLUT = config['path']['trans_lut']
CAMS_PATH = config['path']['trans_lut']
NCPU = config['processor']['ncpu']
NETCDF_ENGINE = config['processor']['netcdf_engine']


@njit
def Gamma2sigma(Gamma):
    '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
    return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)


@njit
def gaussian(x, mu, sigma):
    '''
    Generate gaussian distribution
    :param x:
    :param mu: mode of the Gaussian distribution
    :param sigma: Standard deviation of the Gaussian distribution
    :return:
    '''
    result = np.full((len(x)), np.nan, dtype=np.float32)
    for i in range(len(result)):
        result[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x[i] - mu) ** 2 / (2 * sigma ** 2))
    return result


class LUT:
    def __init__(self,
                 wl=np.arange(350, 2500, 10),
                 lut_file=opj(GRSDATA, TOALUT),
                 trans_lut_file=opj(GRSDATA, TRANSLUT)):
        '''

        :param wl: array of wavelength to process in nm
        :param lut_file: path for diffuse light radiation LUT
        :param trans_lut_file: path for irradiance transmittance LUT
        '''

        # set parameters
        self.wl = wl

        # get path of necessary look-up tables

        self.lut_file = lut_file
        self.trans_lut_file = trans_lut_file
        self.dirdata = config['path']['grsdata']
        self.abs_gas_file = files('radcalnet_oc.data.lut.gases') / 'lut_abs_opt_thickness_normalized.nc'
        # self.lut_file = opj(self.dirdata, 'lut', 'opac_osoaa_lut_v2.nc')
        self.water_vapor_transmittance_file = files('radcalnet_oc.data.lut.gases') / 'water_vapor_transmittance.nc'
        print(self.abs_gas_file)
        self.load_auxiliary_data()

    def load_auxiliary_data(self):
        '''
        Load look-up tables data for gas absorption and backgroud transmittance

        :return:
        '''

        logging.info('loading look-up tables')
        self.trans_lut = xr.open_dataset(self.trans_lut_file, engine=NETCDF_ENGINE)
        # convert wavelength in nanometer
        self.trans_lut['wl'] = self.trans_lut['wl'] * 1000
        self.trans_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'

        self.aero_lut = xr.open_dataset(self.lut_file, engine=NETCDF_ENGINE)
        # convert wavelength in nanometer
        self.aero_lut['wl'] = self.aero_lut['wl'] * 1000
        self.aero_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        self.aero_lut['aot'] = self.aero_lut.aot.isel(wind=0).squeeze()

        self.gas_lut = xr.open_dataset(self.abs_gas_file, engine='h5netcdf')
        self.Twv_lut = xr.open_dataset(self.water_vapor_transmittance_file, engine='h5netcdf')

    def lut_preparation(self,
                        wind=2,
                        sza=[20, 40, 60],
                        vza=[0],
                        azi=[0],
                        aot_refs=np.linspace(0.0, 0.8, 25),
                        weights=[0, 0.5, 1, 0., 0.]):

        logging.info('LUT preparation')

        aero_lut = self.aero_lut.sel(wind=wind, method='nearest')
        trans_lut = self.trans_lut.sel(wind=wind, method='nearest')

        # set mixture of aerosol models
        # ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70']
        for ii, weight in enumerate(weights):
            if ii == 0:
                mix_aero_lut_ = weight * aero_lut.isel(model=ii)
                mix_trans_lut_ = weight * trans_lut.isel(model=ii)
            else:
                mix_aero_lut_ = mix_aero_lut_ + weight * aero_lut.isel(model=ii)
                mix_trans_lut_ = mix_trans_lut_ + weight * trans_lut.isel(model=ii)
        mix_aero_lut_ = mix_aero_lut_ / np.sum(weights)
        mix_trans_lut_ = mix_trans_lut_ / np.sum(weights)

        aero_lut = mix_aero_lut_
        self.trans_aero_lut = mix_trans_lut_
        del mix_aero_lut_, mix_trans_lut_

        #-----------------------------
        # interpolation transmittance
        #-----------------------------
        self.trans_aero_lut = self.trans_aero_lut.interp(sza=[*sza, *vza])
        self.trans_aero_lut = self.trans_aero_lut.interp(aot_ref=aot_refs, method='quadratic')
        self.trans_aero_lut = self.trans_aero_lut.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation Rayleigh
        # -----------------------------
        self.Rray =aero_lut.I.interp(sza=sza, vza=vza).interp(azi=azi).interp(aot_ref=0, method='quadratic')
        self.Rray = self.Rray/np.cos(np.radians(self.Rray.sza))
        self.Rray =self.Rray.interp(wl=self.wl, method='quadratic')

        # -----------------------------
        # interpolation atmo diffuse light
        # -----------------------------
        self.Rdiff_lut = aero_lut.I.interp(sza=sza, vza=vza)
        self.Rdiff_lut =self.Rdiff_lut.interp(azi=azi).interp(aot_ref=aot_refs, method='quadratic')
        self.Rdiff_lut =self.Rdiff_lut /np.cos(np.radians(self.Rdiff_lut.sza))
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
    def __init__(self, wl=None):
        # load data from raw files
        self.solar_irr = SolarIrradiance()
        self.sunglint_eps = pd.read_csv(sunglint_eps_file, sep=r'\s+', index_col=0).to_xarray()
        self.rayleigh()
        self.pressure_rot_ref = 1013.25

        # reproject onto desired wavelengths
        if wl is not None:
            self.solar_irr = self.solar_irr.interp(wl=wl)
            self.sunglint_eps = self.sunglint_eps['mean'].interp(wl=wl)
            self.rot = self.rot.interp(wl=wl)

    def rayleigh(self):
        '''
        Rayleigh Optical Thickness for
        P=1013.25mb,
        T=288.15K,
        CO2=360ppm
        from
        Bodhaine, B.A., Wood, N.B, Dutton, E.G., Slusser, J.R. (1999). On Rayleigh
        Optical Depth Calculations, J. Atmos. Ocean Tech., 16, 1854-1861.
        '''
        data = pd.read_csv(rayleigh_file, skiprows=16, sep=' ', header=None)
        data.columns = ('wl', 'rot', 'dpol')
        self.rot = data.set_index('wl').to_xarray().rot


class SolarIrradiance():
    def __init__(self, wl=None):
        # load data from raw files
        self.wl_min = 300
        self.wl_max = 2600

        self.gueymard = self.read_gueymard()
        self.kurucz = self.read_kurucz()
        self.thuillier = self.read_thuillier()
        self.tsis = self.read_tsis()

    def read_tsis(self):
        '''
        Open TSIS data and convert them into xarray in mW/m2/nm
        :return:
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
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
        '''
        solar_irr = xr.open_dataset(thuillier_file).squeeze().data.drop('time') * 1e3
        solar_irr = solar_irr.rename({'wavelength': 'wl'})
        # keep spectral range of interest UV-SWIR
        solar_irr = solar_irr[(solar_irr.wl <= self.wl_max) & (solar_irr.wl >= self.wl_min)]
        solar_irr.attrs['units'] = 'mW/m2/nm'
        return solar_irr

    def read_gueymard(self):
        '''
        Open Thuillier data and convert them into xarray in mW/m2/nm
        :return:
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
        Open Kurucz data and convert them into xarray in mW/m2/nm
        :return:
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
        Interpolation on new wavelengths
        :param wl: wavelength in nm
        :return: update variables of the class
        '''
        self.thuillier = self.thuillier.interp(wl=wl)
        self.gueymard = self.gueymard.interp(wl=wl)


class Spectral():
    def __init__(self,
                 central_wl,
                 fwhm):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param central_wl: numpy array of the central wavelengths
        :param fwhm: scalar or numpy array containing full width at half maximum in nm                :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''
        self.central_wl = central_wl
        if not isinstance(fwhm, np.ndarray):
            fwhm = np.array([fwhm] * len(central_wl))

        self.fwhm = xr.DataArray(fwhm, name='fwhm',
                                 coords={'wl': central_wl},
                                 attrs={
                                     'definition': 'full width at half maximum of spectral responses modeled as gaussian distributions'})

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.fwhm.groupby('wl'):
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

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
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :return: numpy of convoluted signal
        '''
        Nwl = len(wl)
        signal_ = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = Gamma2sigma(fwhm[ii])
            rsr = gaussian(wl_signal, wl[ii], sig)
            signal_[ii] = np.trapz((signal * rsr), wl_signal) / np.trapz(rsr, wl_signal)
        return signal_

    def convolve(self,
                 signal,
                 name='signal',
                 info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''
        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values

        signal_int = self.convolve_(wl_ref, signal.values, wl, fwhm)
        # signal_int = []
        # for fwhm_ in self.fwhm:
        #     sig = self.Gamma2sigma(fwhm_.values)
        #     rsr = self.gaussian(wl_ref, fwhm_.wl.values, sig)
        #
        #     signal_ = (signal * rsr).integrate('wl') / np.trapz(rsr, wl_ref)
        #     signal_int.append(signal_.values)
        return xr.DataArray(signal_int, name=name,
                            coords={'wl': self.fwhm.wl.values},
                            attrs=info)
