'''
Atmospheric components of the simulation: aerosol model retrieval, gaseous transmittance and
miscellaneous radiometric utilities (see :doc:`/methods`).
'''



import os, sys
from importlib.resources import files

import numpy as np
import xarray as xr

from scipy.optimize import least_squares

import logging

from .config import load_config

opj = os.path.join


# --------------------------------------------------
# settings of the configuration (see radcalnet_oc.config)
# --------------------------------------------------
config = load_config()

AEROSOL_MODELS = config['settings']['aerosol_models']
AEROSOL_COMBINATION = config['settings']['aerosol_combination']
NETCDF_ENGINE = config['processor']['netcdf_engine']
# absorption of O2: 'reptran' (REPTRAN fine table) or 'lbl' (line by line in the O2-B and O2-A bands)
O2_MODELS = ('reptran', 'lbl')
O2_MODEL = config['settings'].get('o2_model', 'reptran')
O2_LBL_FILE = files('radcalnet_oc.data.lut.gases') / 'o2_lbl_tau.nc'
# coupling of the gaseous absorption with the scattering (GasCoupling): 'path_length' (photon path-length model,
# default), 'effective_air_mass' (per-window effective air masses) or 'decoupled' (direct-beam air mass for all
# the terms)
GAS_COUPLINGS = ('decoupled', 'effective_air_mass', 'path_length')
GAS_COUPLING = config['settings'].get('gas_coupling', 'path_length')
GAS_COUPLING_FILE = files('radcalnet_oc.data.auxdata') / 'gas_coupling_eta.csv'
GAS_COUPLING_PPLD_FILE = files('radcalnet_oc.data.auxdata') / 'gas_coupling_ppld.csv'
S2_MIN = 1e-4  # minimum relative variance of the photon path-length distribution


def path_length_transmittance(x, s2):
    r'''
    Transmittance for a Gamma distribution of the absorber optical paths of mean :math:`x` and relative variance
    :math:`s^2` (Laplace transform): :math:`(1 + s^2 x)^{-1/s^2}`, :math:`e^{-x}` when :math:`s^2 \to 0`.

    :param x: mean absorber optical path (:math:`\eta\, m\, \tau`)
    :param s2: relative variance of the distribution (clipped to ``S2_MIN``)
    '''
    s2 = np.maximum(s2, S2_MIN)
    return np.exp(-np.log1p(s2 * x) / s2)

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


class O2LineByLine:
    r'''
    Line-by-line O2 transmittance in the O2-B (683-702 nm) and O2-A (755-775 nm) bands, averaged in the cells
    of a wavelength grid.

    The table ``o2_lbl_tau.nc`` of the package gives the vertical absorption optical thickness :math:`\tau_{O_2}`
    of O2 from the HITRAN lines (Voigt profiles, AFGL midlatitude summer profile, surface pressure
    :math:`P_{lbl}` = 1013 hPa, step 0.004 cm-1; built by ``scripts/make_o2_lbl_lut.py``). The lines are much
    narrower than the cells of the gaseous LUT (0.04-0.06 nm), so the transmittance of a path is averaged in
    each cell, :math:`\langle \exp(-f\,\tau_{O_2}) \rangle` with :math:`f = c_{O_2}\, m\, P/P_{lbl}`, rather than
    computed from a mean optical thickness. Paths combined in a product (e.g. Sun-surface-sensor) must therefore
    be computed with their total air mass, not as the product of one-way transmittances.

    :ivar mask: cells of the grid covered by the table (:py:class:`numpy.ndarray` of booleans)
    '''

    def __init__(self, wl, file=O2_LBL_FILE, block=32):
        '''
        :param wl: wavelengths of the grid (nm), increasing
        :param file: line-by-line optical thickness table
        :param block: number of paths computed at once (memory: block x 187 502 values)
        '''
        with xr.open_dataarray(file) as tau:
            tau = tau.load()
        self.p_ref = float(tau.attrs['surface_pressure'])
        self.attrs = tau.attrs
        wl = np.asarray(wl, dtype=float)
        edges = np.r_[wl[0] - (wl[1] - wl[0]) / 2, 0.5 * (wl[1:] + wl[:-1]), wl[-1] + (wl[-1] - wl[-2]) / 2]
        idx = np.digitize(tau.wl.values, edges) - 1
        ok = (idx >= 0) & (idx < len(wl))
        self.tau = tau.values.astype(float)[ok]
        idx = idx[ok]
        # the table wavelengths are increasing: the points of a cell are contiguous (no cell when the grid
        # does not cover the bands)
        self.starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1] if len(idx) else np.zeros(0, dtype=int)
        self.cells = idx[self.starts]
        self.counts = np.diff(np.r_[self.starts, len(idx)])
        self.mask = np.zeros(len(wl), dtype=bool)
        self.mask[self.cells] = True
        self.nwl = len(wl)
        self.block = block

    def transmittance(self, factor):
        r'''
        Cell-averaged transmittance :math:`\langle \exp(-f\,\tau_{O_2}) \rangle` (1 outside the bands).

        :param factor: :math:`f = c_{O_2}\, m\, P/P_{lbl}`, scalar or array
        :return: array of shape ``factor.shape + (len(wl),)``
        '''
        factor = np.asarray(factor, dtype=float)
        flat = factor.reshape(-1)
        out = np.ones((flat.size, self.nwl))
        for i in range(0, flat.size if len(self.cells) else 0, self.block):
            t = np.exp(-flat[i:i + self.block, None] * self.tau[None, :])
            out[i:i + self.block, self.cells] = np.add.reduceat(t, self.starts, axis=1) / self.counts
        return out.reshape(factor.shape + (self.nwl,))

    def transmittance_path_length(self, factor, eta, s2, tau_add):
        r'''
        Cell-averaged transmittance of the photon path-length model (:py:func:`path_length_transmittance`)
        for the O2 lines plus an absorption constant in each cell (other gases, REPTRAN):
        :math:`\langle (1 + s^2 \eta\, (\tau_c + f \tau_{O_2}))^{-1/s^2} \rangle` in the cells of the bands,
        :math:`(1 + s^2 \eta\, \tau_c)^{-1/s^2}` elsewhere.

        :param factor: :math:`f = c_{O_2}\, m\, P/P_{lbl}`, array of N paths
        :param eta: ratio of the mean path to the geometric path, N values
        :param s2: relative variance of the path distribution, N values
        :param tau_add: optical path of the other absorbers along the geometric path, array (N, len(wl))
        :return: array (N, len(wl))
        '''
        factor, eta, s2 = (np.asarray(v, dtype=float).reshape(-1) for v in (factor, eta, s2))
        tau_add = np.asarray(tau_add, dtype=float).reshape(factor.size, self.nwl)
        out = path_length_transmittance(eta[:, None] * tau_add, s2[:, None])
        if len(self.cells):
            point_cell = np.repeat(self.cells, self.counts)
            for i in range(0, factor.size, self.block):
                b = slice(i, i + self.block)
                x = eta[b, None] * (tau_add[b][:, point_cell] + factor[b, None] * self.tau[None, :])
                t = path_length_transmittance(x, s2[b, None])
                out[b, self.cells] = np.add.reduceat(t, self.starts, axis=1) / self.counts
        return out


class GasCoupling:
    r'''
    Gaseous transmittance of the scattered light, per absorption window: coupling of the absorption with
    the scattering.

    The transmittance :math:`T_g(m)` of the direct beam is exact for the direct light only. The photons
    scattered by the molecules and aerosols above the absorbers shortcut part of the path (less absorption
    for the path reflectance, especially for the water vapour, in the lowest kilometres), while the skylight
    reaching the surface has longer paths than the direct beam (more absorption for the water-leaving term
    and the irradiance). The table ``data/auxdata/gas_coupling_eta.csv``, fitted on coupled
    absorption-scattering simulations (SOS-ABS, correlated k; ``scripts/sos_abs_coupling``), gives for each
    window and term a coefficient

    .. math:: c = c_0 + c_1\,(m - 2) + c_2\,(\tau_{550} - 0.1) + c_3 \ln(U / 2\,\mathrm{g\,cm^{-2}})
       + c_4\, \tau_g

    used in one of two forms (column ``form``):

    * ``'eta'``: effective air mass, :math:`T = T_g(\eta\, m)` with :math:`\eta = c`;
    * ``'blend'``: single-scattering blend, :math:`T = T_g(m)^{1-k}\, T_{ss}^{\,k}` with :math:`k = c` and
      :math:`T_{ss} = (1 - T_g(m)) / (-\ln T_g(m))`, the transmittance seen in single scattering by the light
      scattered uniformly along the absorber column (effective air mass :math:`m/2` for a weak absorption,
      less when the absorption saturates).

    The terms are ``'atm'`` (path reflectance, :math:`m = 1/\mu_0 + 1/\mu_v`), ``'w'`` (water-leaving term,
    same :math:`m`) and ``'Ed'`` (downwelling irradiance, :math:`m = 1/\mu_0`); :math:`U` is the water vapour
    column (:math:`c_3 = 0` in the O2 windows) and :math:`\tau_g = -\ln \bar T_g(m)` the absorption strength
    at the resolution of the fit (blend form only). :math:`m`, :math:`\tau_{550}`, :math:`U` and
    :math:`\tau_g` are clipped to the ranges of the simulations.

    :ivar table: :py:class:`pandas.DataFrame` of the coefficients (columns ``window``, ``wl_min``,
        ``wl_max``, ``term``, ``form``, ``c0``-``c4``, ``m_min``, ``m_max``, ``aot_min``, ``aot_max``,
        ``h2o_min``, ``h2o_max``, ``tau_max``; ``h2o`` in g cm-2)
    '''
    TERMS = ('atm', 'w', 'Ed')
    FORMS = ('eta', 'blend', 'ppld')
    H2O_REF = 2.  # g cm-2
    RESOLUTION = 5.  # cm-1, spectral resolution of the transmittances of the fit (CKD intervals)

    def __init__(self, file=GAS_COUPLING_FILE, resolution=RESOLUTION):
        '''
        :param file: table of the coefficients, file or :py:class:`pandas.DataFrame`
        :param resolution: spectral resolution (cm-1) at which the correction is computed (see
            :py:meth:`correction`); None: channel by channel (not consistent with the fit for the
            saturated channels of a fine table, for tests only)
        '''
        self.resolution = resolution
        import pandas as pd
        if isinstance(file, pd.DataFrame):
            table = file.copy()
        else:
            with open(file) as f:
                table = pd.read_csv(f, comment='#')
        # defaults of the tables without H2O term or form (first version)
        defaults = {'form': 'eta', 'c3': 0., 'c4': 0., 'h2o_min': self.H2O_REF, 'h2o_max': self.H2O_REF,
                    'tau_max': np.inf, 's0': 0., 's1': 0., 's2': 0., 's3': 0.}
        for k, v in defaults.items():
            if k not in table:
                table[k] = v
        if not table.form.isin(self.FORMS).all():
            raise ValueError(f'form must be one of {self.FORMS}')
        self.table = table

    @property
    def windows(self):
        '''Absorption windows: dict ``{name: (wl_min, wl_max)}`` (nm).'''
        t = self.table.drop_duplicates('window')
        return {r.window: (float(r.wl_min), float(r.wl_max)) for r in t.itertuples()}

    def row(self, term, window):
        '''Coefficients of a term and a window (:py:class:`pandas.Series`).'''
        if term not in self.TERMS:
            raise ValueError(f'term must be one of {self.TERMS}, not {term!r}')
        r = self.table[(self.table.window == window) & (self.table.term == term)]
        if len(r) != 1:
            raise KeyError(f'no coefficients for window {window!r} and term {term!r}')
        return r.iloc[0]

    def form(self, term, window):
        '''Form of the correction of a term in a window, ``'eta'`` or ``'blend'``.'''
        return self.row(term, window).form

    def coefficient(self, term, window, air_mass, aot550, h2o=None, tau=None):
        r'''
        Coefficient :math:`c` of a term: :math:`\eta` (form ``'eta'``) or :math:`k` (form ``'blend'``).

        :param term: ``'atm'``, ``'w'`` or ``'Ed'``
        :param window: name of the absorption window (see :py:attr:`windows`)
        :param air_mass: geometric air mass :math:`m` of the term
        :param aot550: aerosol optical thickness at 550 nm
        :param h2o: water vapour column (g cm-2); None: 2 g cm-2
        :param tau: absorption strength :math:`\tau_g = -\ln \bar T_g(m)` (blend form); None: 0
        :return: broadcast of ``air_mass``, ``aot550``, ``h2o`` and ``tau``
        '''
        r = self.row(term, window)
        m = np.clip(air_mass, r.m_min, r.m_max)
        aot = np.clip(aot550, r.aot_min, r.aot_max)
        out = r.c0 + r.c1 * (m - 2) + r.c2 * (aot - 0.1)
        if r.c3 != 0 and h2o is not None:
            out = out + r.c3 * np.log(np.clip(h2o, r.h2o_min, r.h2o_max) / self.H2O_REF)
        if r.c4 != 0 and tau is not None:
            out = out + r.c4 * np.clip(tau, 0., r.tau_max)
        return out

    def path_length(self, term, window, air_mass, aot550, h2o=None):
        r'''
        Parameters of the photon path-length model (form ``'ppld'``): ratio :math:`\eta` of the mean absorber path
        to the geometric one (coefficients ``c0``-``c3``) and relative variance :math:`s^2` of the path
        distribution (coefficients ``s0``-``s3``), both linear in :math:`m - 2`, :math:`\tau_{550} - 0.1` and
        :math:`\ln(U/2)`.

        :return: ``(eta, s2)``, broadcast of ``air_mass``, ``aot550`` and ``h2o``
        '''
        r = self.row(term, window)
        m = np.clip(air_mass, r.m_min, r.m_max)
        aot = np.clip(aot550, r.aot_min, r.aot_max)
        lu = 0. if h2o is None else np.log(np.clip(h2o, r.h2o_min, r.h2o_max) / self.H2O_REF)
        eta = r.c0 + r.c1 * (m - 2) + r.c2 * (aot - 0.1) + r.c3 * lu
        s2 = r.s0 + r.s1 * (m - 2) + r.s2 * (aot - 0.1) + r.s3 * lu
        return eta, np.maximum(s2, S2_MIN)

    def eta(self, term, window, air_mass, aot550, h2o=None):
        r'''Ratio :math:`\eta` of the effective to the geometric air mass (form ``'eta'``, see
        :py:meth:`coefficient`).'''
        if self.form(term, window) != 'eta':
            raise ValueError(f'{term} in {window}: form {self.form(term, window)!r}, not eta')
        return self.coefficient(term, window, air_mass, aot550, h2o)

    @staticmethod
    def smooth(T, resolution=RESOLUTION):
        r'''
        Running mean of a transmittance over ``resolution`` (cm-1) in wavenumber, weighted by the widths
        of the cells of the wavelength grid: transmittance of the intervals of the CKD simulations on which
        the coefficients are fitted.

        :param T: :py:class:`xarray.DataArray` with a ``wl`` dimension (nm, increasing)
        '''
        wl = T.wl.values
        edges = np.r_[wl[0] - (wl[1] - wl[0]) / 2, 0.5 * (wl[1:] + wl[:-1]), wl[-1] + (wl[-1] - wl[-2]) / 2]
        dw = np.diff(edges)
        nu = 1e7 / wl
        lo = np.searchsorted(wl, 1e7 / (nu + resolution / 2), side='left')
        hi = np.searchsorted(wl, 1e7 / (nu - resolution / 2), side='right')
        dims = [d for d in T.dims if d != 'wl'] + ['wl']
        x = T.transpose(*dims)
        c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x.values * dw, axis=-1)], axis=-1)
        w = np.r_[0, np.cumsum(dw)]
        mean = (c[..., hi] - c[..., lo]) / (w[hi] - w[lo])
        return x.copy(data=mean).transpose(*T.dims)

    def correction(self, term, window, Tg_of_m, air_mass, aot550, h2o=None):
        r'''
        Effective transmittance of a term in a window.

        The coefficients are fitted on the transmittances of 5 cm-1 intervals, much less saturated than the
        channels of a fine table (REPTRAN, line by line): applied channel by channel, the blend would
        over-correct the saturated channels. The correction is therefore computed on the transmittances
        smoothed to the resolution of the fit, :math:`\bar T = \langle T_g \rangle_{5\,cm^{-1}}`, and applied
        to each channel so as to keep the interval means, by scaling the absorptance when the absorption
        decreases (path reflectance) and the transmittance when it increases:

        .. math:: T = 1 - (1 - T_g(m))\, \frac{1 - \bar T^{model}}{1 - \bar T(m)} \quad \text{or} \quad
           T = T_g(m)\, \frac{\bar T^{model}}{\bar T(m)},

        with :math:`\bar T^{model} = \langle T_g(\eta m) \rangle` (form ``'eta'``) or
        :math:`\bar T(m)^{1-k}\, T_{ss}(\bar T(m))^k` (form ``'blend'``, :math:`k` with
        :math:`\tau_g = -\ln \bar T(m)`). The band average of the corrected
        transmittance is then that of the simulations whatever the resolution of the table.

        :param term: ``'atm'``, ``'w'`` or ``'Ed'``
        :param window: name of the absorption window
        :param Tg_of_m: function of the air mass returning the transmittance of the coupled gases on the
            wavelengths of the window
        :param air_mass: geometric air mass :math:`m`
        :param aot550: aerosol optical thickness at 550 nm
        :param h2o: water vapour column (g cm-2)
        :return: corrected transmittance of the coupled gases
        '''
        T0 = Tg_of_m(air_mass)
        eta_form = self.form(term, window) == 'eta'
        if eta_form:
            c = self.coefficient(term, window, air_mass, aot550, h2o)
        if self.resolution is None:
            if eta_form:
                return Tg_of_m(c * air_mass)
            c = self.coefficient(term, window, air_mass, aot550, h2o, tau=-np.log(T0))
            return T0 ** (1 - c) * self.single_scattering(T0) ** c
        T0s = self.smooth(T0, self.resolution)
        if eta_form:
            Ts = self.smooth(Tg_of_m(c * air_mass), self.resolution)
        else:
            c = self.coefficient(term, window, air_mass, aot550, h2o, tau=-np.log(T0s))
            Ts = T0s ** (1 - c) * self.single_scattering(T0s) ** c
        # less absorption (path reflectance): absorptance scaled, T0 <= T <= 1; more absorption (water-leaving
        # term, irradiance): transmittance scaled, 0 <= T <= T0; both keep the interval means
        less = Ts >= T0s
        absorptance = ((1 - Ts) / (1 - T0s)).where(T0s < 1., 1.)
        factor = (Ts / T0s).where(T0s > 1e-300, 1.)
        return xr.where(less, 1 - (1 - T0) * absorptance, T0 * factor)

    @staticmethod
    def single_scattering(Tg):
        r'''
        :math:`T_{ss} = (1 - T_g) / (-\ln T_g)`, transmittance averaged over the absorber amounts
        :math:`x\, \tau m` for :math:`x` uniform in [0, 1] (1 for :math:`T_g = 1`).
        '''
        tau = -np.log(Tg)
        small = tau < 1e-12
        return xr.where(small, 1., -np.expm1(-tau) / xr.where(small, 1., tau))


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

    With ``o2_model='lbl'``, the O2 absorption of the O2-B and O2-A bands is computed line by line
    (:py:class:`O2LineByLine`) in place of the REPTRAN table, which is a band parameterization: used as a
    spectrum at sub-nanometre resolution, it makes these bands about twice too deep. Outside these bands
    REPTRAN is used for O2 too.
    '''

    def __init__(self,
                 gas_lut: xr.DataArray,
                 zenith_angle=0,
                 o2_model=None
                 ):
        r'''
        :param gas_lut: look-up table of the normalized absorption optical thickness of the gases
            (:py:class:`xarray.Dataset`, e.g. ``LUT().gas_lut``)
        :param zenith_angle: zenith angle of the path (solar or viewing, deg), used to set the air mass
            :math:`m = 1/\cos\theta`
        :param o2_model: absorption of O2, ``'reptran'`` or ``'lbl'`` (line by line in the O2-B and O2-A
            bands); by default ``settings: o2_model`` of the configuration (``'reptran'``)
        '''
        Gases.__init__(self)
        self.air_mass = 1. / np.cos(np.radians(zenith_angle))
        self.gas_lut = gas_lut
        self.o2_model = O2_MODEL if o2_model is None else o2_model
        if self.o2_model not in O2_MODELS:
            raise ValueError(f'o2_model must be one of {O2_MODELS}, not {self.o2_model!r}')
        self._o2_lbl = None
        self._windows = {}

    @property
    def o2_lbl(self):
        '''Line-by-line O2 on the wavelengths of the gaseous LUT (:py:class:`O2LineByLine`, loaded once).'''
        if self._o2_lbl is None:
            self._o2_lbl = O2LineByLine(self.gas_lut.wl.values)
        return self._o2_lbl

    def Tgas_background(self):
        '''
        Direct transmittance of the background gases (:math:`CO`, :math:`CO_2`, :math:`O_2`, :math:`O_4`),
        with their optical thickness scaled by the ratio of the surface pressure to the reference pressure.

        :return: transmittance at the wavelengths of the gaseous LUT (also stored in ``Tg_bg``)
        '''
        gl = self.gas_lut
        o2 = gl.o2
        if self.o2_model == 'lbl':
            # O2-B and O2-A computed line by line (Tgas_o2_lbl)
            o2 = o2.where(~xr.DataArray(self.o2_lbl.mask, coords={'wl': gl.wl}, dims='wl'), 0.)
        self.ot_air = self.pressure / self.pressure_gas_ref * \
                      (gl.co + self.coef_abs_scat['co2'] * gl.co2 +
                       self.coef_abs_scat['o2'] * o2 +
                       self.coef_abs_scat['o4'] * gl.o4)
        self.Tg_bg = np.exp(- self.air_mass * self.ot_air)
        return self.Tg_bg

    def Tgas_o2_lbl(self):
        r'''
        Line-by-line transmittance of O2 in the O2-B and O2-A bands (1 elsewhere), averaged in the cells of
        the gaseous LUT, for the air mass and the surface pressure of the object (see
        :py:class:`O2LineByLine`).

        :return: transmittance at the wavelengths of the gaseous LUT, :py:class:`xarray.DataArray` with the
            dimensions of ``air_mass`` and ``pressure`` followed by ``wl``
        '''
        lbl = self.o2_lbl
        factor = self.coef_abs_scat['o2'] * self.air_mass * self.pressure / lbl.p_ref
        wl = self.gas_lut.wl
        if isinstance(factor, xr.DataArray):
            return xr.apply_ufunc(lbl.transmittance, factor, output_core_dims=[['wl']]).assign_coords(wl=wl)
        factor = np.asarray(factor, dtype=float)
        if factor.ndim > 0:
            raise ValueError('air_mass and pressure must be scalars or xarray.DataArray with o2_model="lbl"')
        return xr.DataArray(lbl.transmittance(factor), coords={'wl': wl}, dims='wl')

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
                                  background=True,
                                  o2_lbl=True):
        '''
        Total direct transmittance of the absorbing gases, product of the transmittance of each gas.

        :param gases: names of the variable gases to include
        :param background: if True, include the background gases (:py:meth:`Tgas_background`); with
            ``o2_model='lbl'`` also the line-by-line O2 bands (:py:meth:`Tgas_o2_lbl`)
        :param o2_lbl: if False, leave out the line-by-line O2 bands (with ``o2_model='lbl'``)
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
            if self.o2_model == 'lbl' and o2_lbl:
                Tg_tot = Tg_tot * self.Tgas_o2_lbl()

        return Tg_tot

    def get_path_length_transmittance(self, air_mass, eta, s2, gases=('h2o',)):
        r'''
        Transmittance of the given gases and of the background gases for the photon path-length model,
        channel by channel: :math:`(1 + s^2 \eta\, \tau)^{-1/s^2}` with :math:`\tau = -\ln T_g(m)` the optical
        path of the channel along the geometric air mass (a REPTRAN channel is a single exponential); with
        ``o2_model='lbl'``, averaged over the O2 lines of each cell (:py:meth:`O2LineByLine.transmittance_path_length`).

        :param air_mass: geometric air mass :math:`m`
        :param eta: ratio of the mean absorber path to the geometric one
        :param s2: relative variance of the path distribution
        :param gases: variable gases included
        :return: transmittance at the wavelengths of the gaseous LUT
        '''
        self.air_mass = air_mass
        tau = -np.log(self.get_gaseous_transmittance(gases=list(gases), background=True, o2_lbl=False))
        if self.o2_model != 'lbl' or not self.o2_lbl.mask.any():
            return path_length_transmittance(eta * tau, s2)
        lbl = self.o2_lbl
        factor = self.coef_abs_scat['o2'] * self.air_mass * self.pressure / lbl.p_ref
        args = [x if isinstance(x, xr.DataArray) else xr.DataArray(x) for x in (factor, eta, s2)]
        ref = xr.broadcast(tau.isel(wl=0, drop=True), *args)[0]
        dims = list(ref.dims)
        f, e, v = (x.broadcast_like(ref).transpose(*dims) for x in args)
        tau = tau.broadcast_like(ref).transpose(*dims, 'wl')
        out = lbl.transmittance_path_length(f.values, e.values, v.values, tau.values)
        return tau.copy(data=out.reshape(tau.shape))

    # gases whose transmittance is corrected by GasCoupling (with the background gases: O2, O4, CO2, CO);
    # O3 and NO2, mostly above the scatterers, and CH4 keep the direct-beam air mass
    COUPLED_GASES = ['h2o']
    UNCOUPLED_GASES = ['ch4', 'no2', 'o3']

    def _window(self, wl_min, wl_max):
        """GaseousTransmittance on the wavelengths of an absorption window (built once)."""
        key = (wl_min, wl_max)
        if key not in self._windows:
            self._windows[key] = GaseousTransmittance(self.gas_lut.sel(wl=slice(wl_min, wl_max)),
                                                      o2_model=self.o2_model)
        return self._windows[key]

    def get_coupled_transmittance(self, air_mass, term, aot550, coupling):
        r'''
        Effective gaseous transmittance of a term of the signal, with the absorption-scattering coupling.

        In each absorption window of ``coupling``, the transmittance of the water vapour and of the background
        gases (O2...) is corrected as given by ``coupling``: photon path-length model channel by channel
        (form ``'ppld'``, :py:meth:`get_path_length_transmittance`), or effective air mass :math:`\eta\, m` or
        single-scattering blend at the resolution of the fit (:py:meth:`GasCoupling.correction`), with the
        water vapour column ``gas_tc['h2o']``;
        elsewhere, and for the other gases, the air mass :math:`m` is used, as in
        :py:meth:`get_gaseous_transmittance`. ``pressure`` and ``gas_tc`` are set before the call.

        :param air_mass: geometric air mass :math:`m` of the term
        :param term: ``'atm'``, ``'w'`` or ``'Ed'`` (see :py:class:`GasCoupling`)
        :param aot550: aerosol optical thickness at 550 nm
        :param coupling: :py:class:`GasCoupling` object
        :return: transmittance at the wavelengths of the gaseous LUT
        '''
        self.air_mass = air_mass
        Tg = self.get_gaseous_transmittance()
        h2o = self.gas_tc['h2o'] / 10.  # kg m-2 -> g cm-2
        parts = []
        for name, (wl_min, wl_max) in coupling.windows.items():
            sub = self._window(wl_min, wl_max)
            sub.pressure, sub.pressure_gas_ref = self.pressure, self.pressure_gas_ref
            sub.gas_tc, sub.coef_abs_scat = self.gas_tc, self.coef_abs_scat
            sub.air_mass = air_mass
            Tg_other = sub.get_gaseous_transmittance(gases=self.UNCOUPLED_GASES, background=False)

            def Tg_coupled_of(m, sub=sub):
                sub.air_mass = m
                return sub.get_gaseous_transmittance(gases=self.COUPLED_GASES, background=True)

            if coupling.form(term, name) == 'ppld':
                eta, s2 = coupling.path_length(term, name, air_mass, aot550, h2o)
                Tg_coupled = sub.get_path_length_transmittance(air_mass, eta, s2, gases=self.COUPLED_GASES)
            else:
                Tg_coupled = coupling.correction(term, name, Tg_coupled_of, air_mass, aot550, h2o)
            parts.append(Tg_other * Tg_coupled)
        self.air_mass = air_mass
        if not parts:
            return Tg
        out = xr.concat(parts, 'wl').combine_first(Tg)
        return out.transpose(*[d for d in Tg.dims if d in out.dims], ...)


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