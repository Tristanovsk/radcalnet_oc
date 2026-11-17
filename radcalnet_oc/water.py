'''
Optical properties of pure water and spectral extension of the remote-sensing reflectance.

The water-leaving signal is measured (AERONET-OC) or reconstructed (:py:mod:`radcalnet_oc.reconstruction`) over a
limited spectral range, typically 350-1150 nm. :py:func:`extend_rrs` extends it to the whole range of the simulation
(350-2500 nm) with a semi-analytical model in which the absorption is that of pure water, fitted on the near-infrared
end of the input, instead of setting it to zero.
'''

import logging
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# pure water absorption and scattering coefficients (m-1), 200-2449 nm: Smith and Baker (1981) below 380 nm,
# Pope and Fry (1997) 380-700 nm, Kou et al. (1993) above (NASA OCSSW table water_coef.txt)
WATER_COEF_FILE = files('radcalnet_oc.data.auxdata') / 'water_coef.txt'

# coefficients of the relation between rrs (below surface) and u = bb / (a + bb), Gordon et al. (1988) as in
# Lee et al. (2002): rrs = g0 u + g1 u^2
G0, G1 = 0.0895, 0.1247
# above / below surface conversion, Lee et al. (2002): Rrs = 0.52 rrs / (1 - 1.7 rrs)
T_SURF, GAMMA = 0.52, 1.7

# spectral window (nm) of the input on which the particulate backscattering is fitted: the end of the input,
# at most RRS_FIT_WIDTH nm wide, above RRS_FIT_MIN where the absorption of the water constituents is neglected
RRS_FIT_MIN = 750.
RRS_FIT_WIDTH = 100.
# bounds of the factor joining the model to the last values of the input
RRS_JOIN_BOUNDS = (0.5, 2.)


class PureWater:
    '''
    Absorption :math:`a_w` and backscattering :math:`b_{bw} = b_w/2` coefficients of pure water (m-1), at the
    reference temperature of the tables (no temperature nor salinity correction).
    '''

    def __init__(self, file=WATER_COEF_FILE):
        with open(file) as f:
            skip = next(i for i, line in enumerate(f) if line.startswith('/end_header')) + 1
        table = pd.read_csv(file, skiprows=skip, sep=r'\s+', header=None, names=('wl', 'aw', 'bw'))
        self.table = table.set_index('wl').to_xarray()

    def coefficients(self, wl):
        '''
        Pure water coefficients at the wavelengths ``wl`` (nm), interpolated linearly; beyond the last
        wavelength of the table (2449 nm), :math:`\\ln a_w` is extrapolated linearly from its last 50 nm and
        :math:`b_{bw}` is that of the last wavelength.

        :param wl: wavelengths (nm), array or :py:class:`xarray.DataArray`
        :return: ``aw``, ``bbw`` (m-1), :py:class:`xarray.DataArray` with a ``wl`` dimension
        '''
        wl = np.asarray(wl, dtype=float)
        tab = self.table
        last = float(tab.wl[-1])
        aw = tab.aw.interp(wl=wl)
        bbw = tab.bw.interp(wl=wl).fillna(float(tab.bw[-1])) / 2.
        beyond = wl > last
        if beyond.any():
            end = tab.aw.sel(wl=slice(last - 50, last))
            slope, intercept = np.polyfit(end.wl.values, np.log(end.values), 1)
            aw = aw.where(~xr.DataArray(beyond, dims='wl', coords={'wl': wl}),
                          np.exp(intercept + slope * wl))
        return aw.rename('aw'), bbw.rename('bbw')


def rrs_to_u(Rrs):
    '''
    Ratio :math:`u = b_b/(a+b_b)` from the remote-sensing reflectance above the surface (sr-1), inverting
    :math:`r_{rs} = R_{rs}/(0.52 + 1.7 R_{rs})` and :math:`r_{rs} = g_0 u + g_1 u^2` (Lee et al., 2002).
    Negative reflectances (noise) give :math:`u = 0`.
    '''
    rrs = Rrs / (T_SURF + GAMMA * Rrs)
    rrs = rrs.clip(min=0)
    return (-G0 + np.sqrt(G0 ** 2 + 4 * G1 * rrs)) / (2 * G1)


def u_to_rrs(u):
    '''Remote-sensing reflectance above the surface (sr-1) from :math:`u = b_b/(a+b_b)` (inverse of
    :py:func:`rrs_to_u`).'''
    rrs = G0 * u + G1 * u ** 2
    return T_SURF * rrs / (1 - GAMMA * rrs)


def extend_rrs(Rrs,
               wl,
               method='pure_water',
               fit_min=RRS_FIT_MIN,
               fit_width=RRS_FIT_WIDTH):
    r'''
    Interpolate the remote-sensing reflectance at the wavelengths of the simulation and extend it beyond the
    spectral range of the input.

    Within the range of each spectrum, ``Rrs`` is interpolated linearly (also across its missing values).
    Beyond its last wavelength :math:`\lambda_{max}`, with ``method='pure_water'`` (:ref:`methods-rrs`):

    - the particulate backscattering :math:`b_{bp}` is retrieved on the last ``fit_width`` nm of the input
      above ``fit_min`` nm, where the absorption is assumed to be that of pure water,
      :math:`b_{bp} = \langle a_w u/(1-u) - b_{bw} \rangle`, spectrally flat and positive;
    - the reflectance is computed with :math:`u = (b_{bw} + b_{bp})/(a_w + b_{bw} + b_{bp})` and multiplied by
      the factor (bounded to 0.5-2) joining it to the mean of the last 10 nm of the input.

    Below the first wavelength of the input, the first value is kept. With ``method='zero'``, the reflectance
    is set to zero outside the range of the input (as before version 1.1).

    :param Rrs: remote-sensing reflectance above the surface (sr-1), :py:class:`xarray.DataArray` with a ``wl``
        dimension (nm)
    :param wl: wavelengths of the simulation (nm)
    :param method: ``'pure_water'`` or ``'zero'``
    :param fit_min: lower bound (nm) of the window of the fit of :math:`b_{bp}`
    :param fit_width: maximal width (nm) of the window of the fit of :math:`b_{bp}`
    :return: ``Rrs`` at the wavelengths ``wl`` and the particulate backscattering :math:`b_{bp}` (m-1) fitted
        for each spectrum (NaN with ``method='zero'``)
    '''
    if method not in ('pure_water', 'zero'):
        raise ValueError(f"unknown Rrs extension method {method!r}: 'pure_water' or 'zero'")
    wl = np.asarray(wl, dtype=float)
    Rrs = Rrs.sortby('wl')
    # ranges of each spectrum, from its valid values
    valid = Rrs.notnull()
    wl_in = Rrs.wl.where(valid)
    wl_min, wl_max = wl_in.min('wl'), wl_in.max('wl')
    filled = Rrs.interpolate_na('wl')
    out = filled.interp(wl=wl)
    inside = (out.wl >= wl_min) & (out.wl <= wl_max)
    out = out.where(inside)
    other_dims = [dim for dim in Rrs.dims if dim != 'wl']
    nan = xr.full_like(wl_max, np.nan, dtype=float)

    if method == 'zero':
        return out.fillna(0), nan.rename('bbp_nir')

    # below the range: first value of the spectrum
    first = filled.sel(wl=wl_min.fillna(float(Rrs.wl[0])), method='nearest').reset_coords(drop=True)
    out = out.where(out.wl >= wl_min, first)

    # fit of bbp on the near-infrared end of the input
    water = PureWater()
    aw_in, bbw_in = water.coefficients(Rrs.wl.values)
    aw_in, bbw_in = aw_in.assign_coords(wl=Rrs.wl), bbw_in.assign_coords(wl=Rrs.wl)
    lower = np.maximum(wl_max - fit_width, fit_min)
    narrow = (wl_max < fit_min + 10)
    if narrow.any():
        logger.warning(f'Rrs ends below {fit_min + 10:g} nm for {int(narrow.sum())} spectrum(s): particulate '
                       f'backscattering fitted on the last {fit_width:g} nm, with the absorption of pure water')
    lower = lower.where(~narrow, wl_max - fit_width)
    window = (Rrs.wl >= lower) & (Rrs.wl <= wl_max) & valid
    u = rrs_to_u(Rrs).clip(max=0.5)
    bbp = (aw_in * u / (1 - u) - bbw_in).where(window).mean('wl').clip(min=0)

    # model beyond the range, joined to the last 10 nm of the input
    beyond = out.wl > wl_max
    aw, bbw = water.coefficients(wl)
    aw, bbw = aw.assign_coords(wl=wl), bbw.assign_coords(wl=wl)
    bb = bbw + bbp
    model = u_to_rrs(bb / (aw + bb))
    end = (Rrs.wl >= wl_max - 10) & valid
    rrs_end = Rrs.where(end).mean('wl')
    model_end = u_to_rrs((bbw_in + bbp) / (aw_in + bbw_in + bbp)).where(end).mean('wl')
    join = (rrs_end / model_end).where(model_end > 0).clip(*RRS_JOIN_BOUNDS).fillna(1)
    out = out.where(~beyond, join * model)

    # spectra without any valid value
    out = out.where(wl_max.notnull())
    bbp = bbp.where(wl_max.notnull())
    bbp.attrs = {'unit': 'm-1', 'description': 'particulate backscattering coefficient fitted on the near-infrared '
                                               'end of Rrs (absorption of pure water), used beyond its range'}
    out = out.transpose(*other_dims, 'wl') if other_dims else out
    return out, bbp.rename('bbp_nir')
