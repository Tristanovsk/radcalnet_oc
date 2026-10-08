'''
Atmospheric components of the simulation: aerosol model retrieval, gaseous transmittance and
miscellaneous radiometric utilities (see :doc:`/methods`).
'''



import os, sys
import numpy as np
import xarray as xr

from scipy.optimize import least_squares

import yaml
from importlib.resources import files
import logging

opj = os.path.join


# ------------------------------------
# get path of packaged files
# ------------------------------------
LUT_FILE_BACKUP = files('radcalnet_oc.data.lut.atmo').joinpath('toa_lut_opac_ultra_light.nc')

# --------------------------------------------------
# get path of other files as indicated in config.yml
# --------------------------------------------------
configfile = files(__package__) / 'config.yml'
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

LUTDATA = config['path']['lutdata']
TOALUT = config['path']['toa_lut']
AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']
NETCDF_ENGINE = config['processor']['netcdf_engine']

class Aerosol:
    '''
    Retrieval of the proportions of the aerosol models from the spectral aerosol optical thickness (AOT).

    The normalized AOT measured (e.g. by AERONET) is fitted with a mixture of the normalized AOT spectra of
    three OPAC models (desert ``DESE_rh70``, maritime clean ``MACL_rh70`` and water-soluble ``WASO_rh0``)
    by bounded least squares (see :ref:`methods-aerosol`).

    Example::

        aerosol = Aerosol(naot, process.lut.naot_lut)
        aerosol.process()
        process.execute(aerosol_combination=aerosol.model_db.aerosol_combination)

    :ivar model_db: :py:class:`xarray.Dataset` set by :py:meth:`process`, with ``aerosol_combination``
        (dimensions ``time``, ``model``) and the residual ``cost`` of the fit
    '''

    def __init__(self,
                 naot_db,
                 naot_lut,
                 wl_dimension="wl_aeronet",
                 ):
        r'''
        :param naot_db: normalized aerosol optical thickness
            :math:`\tau_a(\lambda)/\tau_a(\lambda_{ref})`, :py:class:`xarray.DataArray` with a ``time``
            dimension and a spectral dimension; missing values are interpolated along the wavelengths and time
        :param naot_lut: normalized aerosol optical thickness of the aerosol models (dimensions ``model``,
            ``wl``), e.g. ``LUT().naot_lut``
        :param wl_dimension: name of the spectral dimension of ``naot_db``
        '''

        self.naot_db = naot_db.rename({wl_dimension: "wl"})
        self.wl = self.naot_db.wl

        # replace NaN for further least_square optimization
        self.naot_db = self.naot_db.interpolate_na('wl')
        if "time" in self.naot_db.dims:
            self.naot_db = self.naot_db.interpolate_na('time')

        # get normalized aot for aerosol model fitting
        self.naot_lut = naot_lut
        self.naot_lut_all = naot_lut_all = self.naot_lut.interp(wl=self.wl, method='quadratic')

        # naot_lut is used to fit the proportion of each model
        self.naot_lut = np.array([naot_lut_all.sel(model="DESE_rh70"),
                                  naot_lut_all.sel(model="MACL_rh70"),
                                  naot_lut_all.sel(model="WASO_rh0")])


    def func_aero(self, fcoef, n_aot):
        '''
        Normalized AOT spectrum of a mixture of the three aerosol models.

        :param fcoef: proportions of the three models (normalized to a sum of 1)
        :param n_aot: normalized AOT spectra of the three models, array of shape (3, number of wavelengths)
        :return: normalized AOT spectrum of the mixture
        '''
        fcoef = fcoef / np.sum(fcoef)
        sim = fcoef[0] * n_aot[0] + fcoef[1] * n_aot[1] + fcoef[2] * n_aot[2]
        return sim

    def cost_func(self,
                  fcoef,
                  n_aot_lut,
                  naot_mes):
        '''
        Residuals of the fit of the aerosol models.

        :param fcoef: proportions of the three models
        :param n_aot_lut: normalized AOT spectra of the three models
        :param naot_mes: measured normalized AOT spectrum
        :return: measured minus simulated normalized AOT, for each wavelength
        '''
        sim = self.func_aero(fcoef, n_aot_lut)
        return naot_mes - sim

    def process(self
                ):
        '''
        Retrieve the proportion of each aerosol model for each time of ``naot_db``.

        The proportions are bounded between 0 and 1 and normalized to a sum of 1; the models not fitted
        are set to 0. The result is stored in ``model_db``, with the models of ``config.yml``
        (``settings: aerosol_models``) as ``model`` coordinate.
        '''


        xarr = []
        for time, naot in self.naot_db.groupby('time'):
            naot = naot.squeeze()
            p0 = [0., 0.5, 0.5]
            res = least_squares(self.cost_func, p0,
                                args=(self.naot_lut, naot),
                                bounds=([0, 0, 0], [1, 1, 1]))
            res.x = res.x / np.sum(res.x)
            # results are time, the proportion of each aerosol model, array ends with the remaining cost
            xarr.append([time, 0, 0, res.x[0], res.x[1], 0, res.x[2], res.cost])
            # xarr.append([time,0,0,0,1,0,0,res.cost])
        xarr = np.array(xarr)

        self.model_db = xr.Dataset(data_vars=dict(aerosol_combination=(['time', 'model'], xarr[:, 1:-1].astype(float)),
                                          cost=('time', xarr[:, -1].astype(float)),
                                          ),
                           coords={'time': xarr[:, 0],
                                   'model': AEROSOL_MODELS})


class CamsParams:
    '''
    Name and resolution of a CAMS parameter.

    :param name: name of the CAMS variable
    :param resol: spatial resolution
    '''

    def __init__(self,
                 name,
                 resol):
        self.name = name
        self.resol = resol


class Gases():
    '''
    Default parameters of the absorbing gases, used by :py:class:`~radcalnet_oc.kernel.GaseousTransmittance`.

    :ivar pressure: surface pressure (hPa), 1010 by default
    :ivar pressure_gas_ref: reference pressure of the background gases optical thickness (hPa)
    :ivar gas_tc: total column of each gas (kg m-2), keys ``'co2'``, ``'o2'``, ``'o4'``, ``'ch4'``,
        ``'no2'``, ``'o3'``, ``'h2o'``
    :ivar coef_abs_scat: scaling coefficient :math:`c_g` of the optical thickness of each gas (1 by default)
    '''

    def __init__(self):
        '''
        Set the default parameters.
        '''

        self.pressure = 1010
        self.pressure_gas_ref = 1000

        self.gas_tc = {'co2': 1,
                       'o2': 1,
                       'o4': 1,
                       'ch4': 1e-2,
                       'no2': 3e-6,
                       'o3': 6.5e-3,
                       'h2o': 30}
        self.coef_abs_scat = {'co2': 1.,
                              'o2': 1.,
                              'o4': 1.,
                              'ch4': 1.,
                              'no2': 1,
                              'o3': 1,
                              'h2o': 1.}


class GaseousTransmittance(Gases):
    r'''
    Direct transmittance of the absorbing gases (see :doc:`/methods`).

    The transmittance of a gas :math:`g` of total column :math:`U_g` is
    :math:`T_g = \exp(-m\, c_g\, U_g\, \kappa_g(\lambda))`, with :math:`\kappa_g` the normalized
    optical thickness of the gaseous look-up table and :math:`m` the air mass of the path. The
    attributes ``pressure``, ``gas_tc`` and ``air_mass`` are set before calling
    :py:meth:`get_gaseous_transmittance`::

        gas_trans = GaseousTransmittance(lut.gas_lut)
        gas_trans.pressure = 1013.
        gas_trans.gas_tc['h2o'] = 25.        # kg m-2
        gas_trans.air_mass = 1 / mu0
        Tg = gas_trans.get_gaseous_transmittance()
    '''

    def __init__(self,
                 gas_lut: xr.DataArray,
                 zenith_angle=0
                 ):
        r'''
        :param gas_lut: look-up table of the normalized absorption optical thickness of the gases
            (:py:class:`xarray.Dataset`, e.g. ``LUT().gas_lut``)
        :param zenith_angle: zenith angle of the path (solar or viewing, deg), used to set the air mass
            :math:`m = 1/\cos\theta`
        '''
        Gases.__init__(self)
        self.air_mass = 1. / np.cos(np.radians(zenith_angle))
        self.gas_lut = gas_lut

    def Tgas_background(self):
        '''
        Direct transmittance of the background gases (:math:`CO`, :math:`CO_2`, :math:`O_2`, :math:`O_4`),
        with their optical thickness scaled by the ratio of the surface pressure to the reference pressure.

        :return: transmittance at the wavelengths of the gaseous LUT (also stored in ``Tg_bg``)
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
        Direct transmittance of one absorbing gas, at the full spectral resolution of the gaseous LUT.

        :param gas_name: name of the gas: ``'h2o'``, ``'o3'``, ``'no2'`` or ``'ch4'``
        :return: transmittance at the wavelengths of the gaseous LUT
        '''

        ot = self.coef_abs_scat[gas_name] * self.gas_tc[gas_name] * self.gas_lut[gas_name]
        Tg = np.exp(- self.air_mass * ot)
        return Tg

    def get_gaseous_transmittance(self,
                                  gases=['ch4', 'no2', 'o3', 'h2o'],
                                  background=True):
        '''
        Total direct transmittance of the absorbing gases, product of the transmittance of each gas.

        :param gases: names of the variable gases to include
        :param background: if True, include the background gases (:py:meth:`Tgas_background`)
        :return: transmittance at the wavelengths of the gaseous LUT
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
    Miscellaneous utilities.
    '''

    @staticmethod
    def get_pressure(alt, psl):
        r'''
        Pressure at a given altitude, from the barometric formula
        :math:`P(z) = P_{sl}\,(1 - 0.0065\, z / 288.15)^{5.255}`.

        :param alt: altitude (m), float or array (NaN are set to 0)
        :param psl: pressure at sea level (hPa)
        :return: pressure at the given altitude (hPa)
        '''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    @staticmethod
    def transmittance_dir(aot, air_mass, rot=0):
        r'''
        Direct (beam) transmittance :math:`\exp\left[-(\tau_r + \tau_a)\, m\right]`.

        :param aot: aerosol optical thickness
        :param air_mass: air mass of the path
        :param rot: Rayleigh optical thickness
        :return: direct transmittance
        '''
        return np.exp(-(rot + aot) * air_mass)

    @staticmethod
    def air_mass(sza, vza):
        r'''
        Air mass of the Sun-surface-sensor path, :math:`1/\cos\theta_v + 1/\cos\theta_s`.

        :param sza: solar zenith angle (deg)
        :param vza: viewing zenith angle (deg)
        :return: air mass
        '''
        return 1 / np.cos(np.radians(vza)) + 1 / np.cos(np.radians(sza))

    @staticmethod
    def earth_sun_correction(dayofyear):
        r'''
        Earth-Sun distance correction factor :math:`d^2 = (\bar{d}/d)^2` of the mean solar irradiance.

        :param dayofyear: day of year (1--366)
        :return: correction factor, to multiply the solar irradiance at mean Earth-Sun distance by
        '''
        theta = 2. * np.pi * dayofyear / 365
        d2 = 1.00011 + 0.034221 * np.cos(theta) + 0.00128 * np.sin(theta) + \
             0.000719 * np.cos(2 * theta) + 0.000077 * np.sin(2 * theta)
        return d2


class Radiometry():
    '''
    Radiometric conversions.
    '''

    def __init__(self):
        # ---------------------------------------------
        #      PARAMETERS
        # Planck constant in J s or W s2
        '''
        Set the product of the Avogadro number, the Planck constant and the speed of light.
        '''
        h = 6.6260695729e-3  # d-34
        # light speed in m s-1
        c = 2.99792458e0  # d8
        # Avogadro Number in mol-1
        Avogadro = 6.0221412927e0  # d23
        self.Ahc = Avogadro * h * c
        # ---------------------------------------------

    def PAR(self,
            Ed: xr.Dataset):
        r'''
        Instantaneous photosynthetically available radiation from a downwelling irradiance spectrum,
        :math:`\mathrm{PAR} = \frac{1}{N_A h c}\int_{400}^{700} \lambda\, E_d(\lambda)\, d\lambda`.

        Typical values above the surface: 1500--2000 µmol photons m-2 s-1 in full sunlight,
        200--500 µmol photons m-2 s-1 on an overcast day or in the morning or evening.

        :param Ed: downwelling irradiance (mW m-2 nm-1), :py:class:`xarray.DataArray` with a ``wl``
            dimension (nm)
        :return: PAR (µmol photons m-2 s-1), numpy array
        '''

        wl_range = slice(400, 700)
        Ed_par = Ed.sel(wl=wl_range).squeeze()

        #
        # Ed_par.integrate('wl').values

        # in mol photon m-2 s-1
        PAR_d = ((Ed_par.wl * Ed_par).integrate('wl') / self.Ahc)
        # conversion in µmol photon m-2 s-1
        PAR_d = 1e-6 * PAR_d
        return PAR_d.values