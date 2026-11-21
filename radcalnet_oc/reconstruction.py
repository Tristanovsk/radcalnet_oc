'''
Hyperspectral reconstruction of the remote-sensing reflectance from multispectral measurements
(e.g. the AERONET-OC bands), with the semi-analytical model of the `invRrs` package.

The settings are those of the AERONET-OC chain of `invRrs` (tutorial ``invRrs_aeronet-oc_chain``). For each
spectrum, the concentrations and optical properties of the water constituents (phytoplankton,
coloured dissolved and detrital matter, mineral particles of two size modes) are retrieved by fitting the
forward model of :py:class:`invRrs.solver.two_modes` to the measured bands, then the forward model gives
the reflectance at any wavelength of 350-1150 nm (range of the sediment optical properties of `invRrs`); beyond,
:py:func:`radcalnet_oc.water.extend_rrs` extends it in :py:class:`~radcalnet_oc.Process`. `invRrs` is an optional
dependency (``pip install invRrs``).
'''

import logging
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# parameters of the invRrs two_modes model, in the order of its parameter vector
PARAMETERS = ('chl', 'a_bg_ref', 'S_bg', 'spm_norm', 'fine_prop', 'nr', 'ni_ref')
PARAMETER_UNITS = ('mg m-3', 'm-1', 'nm-1', '-', '-', '-', '-')
# Chl-specific absorption and backscattering of phytoplankton used by default
PHYTO_MODEL = 'Ciotti2002_Sf03'
# measurement geometry of the AERONET-OC radiometers (deg): viewing zenith and relative azimuth angles
AERONET_OC_GEOMETRY = (40., 90.)
# initial values of the parameters of each fit (chl: AERONET-OC chlorophyll-a when given)
XINIT = dict(chl=1., a_bg_ref=0.05, S_bg=0.017, spm_norm=0.3, fine_prop=0.5, nr=1.17, ni_ref=-0.003)
# upper wavelength (nm) of the sediment optical properties of invRrs
WL_MAX = 1150.
# spectral shape of the imaginary refractive index of the sediments (ni_shape of invRrs), pinned so that the
# reconstruction does not follow the default of the installed invRrs ('wl_a_sed' in 1.4.0, 'atd_ext:0.5' after)
NI_SHAPE = 'atd_ext:0.5'


def _import_invrrs():
    try:
        from invRrs import solver, utils
    except ImportError as error:
        raise ImportError('the hyperspectral reconstruction needs the invRrs package (pip install invRrs)') \
            from error
    return solver, utils


def invrrs_version():
    '''
    Version of the installed invRrs, with the git commit for a development installation (e.g.
    ``1.4.0+g60835a4.dirty``), or ``'not installed'``.
    '''
    try:
        import invRrs
    except ImportError:
        return 'not installed'
    version = getattr(invRrs, '__version__', 'unknown')
    root = Path(invRrs.__file__).resolve().parent.parent
    if not (root / '.git').exists():
        return version
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip()
        commit = git('rev-parse', '--short', 'HEAD')
        dirty = '.dirty' if git('status', '--porcelain', '--untracked-files=no', '--', 'invRrs') else ''
    except (OSError, subprocess.SubprocessError):
        return version
    return f'{version}+g{commit}{dirty}'


def _supported_ni_shape(shape):
    '''``shape`` when the installed invRrs knows it, None (default of invRrs) otherwise, with a warning.'''
    if shape is None:
        return None
    try:
        from invRrs.forward import _check_ni_shape
    except ImportError:
        logger.warning(f'invRrs {invrrs_version()} cannot check ni_shape={shape!r}: its default sediment shape is '
                       f'used')
        return None
    try:
        _check_ni_shape(shape)
    except ValueError:
        logger.warning(f'ni_shape={shape!r} is not available in invRrs {invrrs_version()}: its default sediment '
                       f'shape is used (see the attribute invrrs_ni_shape of the output)')
        return None
    return shape


class Reconstruction:
    '''
    Hyperspectral reconstruction of multispectral remote-sensing reflectances with `invRrs`.

    :param wl: wavelengths (nm) of the reconstructed reflectance, at most 1150 nm
    :param vza: viewing zenith angle (deg) of the measurements (40 deg for AERONET-OC)
    :param azi: relative azimuth angle (deg) of the measurements (90 deg for AERONET-OC)
    :param phyto_model: Chl-specific inherent optical properties of phytoplankton
        (``invRrs.utils.data.load_iop_phyto``)
    :param ni_shape: spectral shape of the imaginary refractive index of the sediments (``ni_shape`` of
        ``invRrs.inverse.TwoModes``), ``NI_SHAPE`` by default; None follows the default of the installed invRrs.
        A shape unknown to the installed invRrs falls back to its default, with a warning
    :param sigma: uncertainty of the measured reflectance weighting the fit, function of the
        reflectance of the bands (e.g. ``lambda rrs: 0.05 * rrs + 2e-5``), or None for an unweighted fit
        (default, as the AERONET-OC chain of invRrs)
    :param max_nfev: maximal number of evaluations of the model per spectrum (scipy convention)
    :param max_sza: spectra with a larger solar zenith angle (deg) are not reconstructed
    :param min_bands: spectra with fewer valid bands are not reconstructed
    '''

    def __init__(self,
                 wl=np.arange(350., WL_MAX + 1, 1.),
                 vza=AERONET_OC_GEOMETRY[0],
                 azi=AERONET_OC_GEOMETRY[1],
                 phyto_model=PHYTO_MODEL,
                 ni_shape=NI_SHAPE,
                 sigma=None,
                 max_nfev=100,
                 max_sza=80.,
                 min_bands=5):
        self.solver, utils = _import_invrrs()
        self.wl = np.asarray(wl, dtype=float)
        if self.wl.max() > WL_MAX:
            raise ValueError(f'the reconstruction is limited to {WL_MAX:g} nm (sediment optical properties of '
                             f'invRrs); beyond, see radcalnet_oc.water.extend_rrs')
        self.vza, self.azi = vza, azi
        self.phyto_model = phyto_model
        self.aphy_star, self.bbphy_star = utils.data().load_iop_phyto(phyto_model)
        self.ni_shape = _supported_ni_shape(ni_shape)
        self.sigma = sigma
        self.max_nfev = max_nfev
        self.max_sza = max_sza
        self.min_bands = min_bands
        # invRrs models, one per wavelength grid (loading the look-up tables is expensive)
        self._models = {}

    def _model(self, wl, sza):
        key = tuple(np.round(wl, 3))
        if key not in self._models:
            shape = {} if self.ni_shape is None else {'ni_shape': self.ni_shape}
            self._models[key] = self.solver.two_modes(self.aphy_star, self.bbphy_star, sza, wl_=np.asarray(wl),
                                                      **shape)
        return self._models[key]

    def provenance(self, model=None):
        '''
        Settings of the reconstruction, written as global attributes of the output of :py:meth:`reconstruct`:
        version of invRrs (:py:func:`~radcalnet_oc.reconstruction.invrrs_version`), sediment n_i shape actually
        used, phytoplankton optical properties, fluorescence term actually fitted, weighting, geometry of the
        measurements and limits of the fits.

        :param model: an invRrs model of this reconstruction, to read the n_i shape it uses
        :return: dict of text and number attributes (NetCDF compatible)
        '''
        shape = getattr(model, 'ni_shape', None) if model is not None else self.ni_shape
        return {'invrrs_version': invrrs_version(),
                'invrrs_model': 'TwoModes',
                'invrrs_ni_shape': str(shape) if shape is not None else 'invRrs default',
                'invrrs_phyto_model': str(self.phyto_model),
                'invrrs_fluorescence': (f"{getattr(model, 'fluo_model', '?')} ({model.fit_fluo})"
                                        if getattr(model, 'fit_fluo', False) else 'none'),
                'invrrs_sigma': 'none' if self.sigma is None else 'user function',
                'invrrs_max_nfev': int(self.max_nfev),
                'invrrs_vza': float(self.vza),
                'invrrs_azi': float(self.azi),
                'invrrs_max_sza': float(self.max_sza),
                'invrrs_min_bands': int(self.min_bands)}

    def reconstruct(self,
                    Rrs,
                    sza,
                    chl=None,
                    band_dim='wl'):
        '''
        Reconstruct the hyperspectral reflectance of each spectrum of a time series.

        Each fit starts from ``XINIT``, with ``chl`` as initial chlorophyll concentration when given; negative
        reflectances are set to zero. The reflectance is computed at the geometry of the measurements
        (``Rrs``) and at nadir (``Rrs_nadir``).

        :param Rrs: multispectral reflectance (sr-1), :py:class:`xarray.DataArray` with a ``time`` dimension
            and the dimension ``band_dim`` of the central wavelengths (nm); missing bands are left out
        :param sza: solar zenith angles (deg), :py:class:`xarray.DataArray` with a ``time`` dimension
        :param chl: chlorophyll-a concentration (mg m-3, e.g. the AERONET-OC product), first guess
            of each fit, :py:class:`xarray.DataArray` with a ``time`` dimension or None
        :param band_dim: name of the dimension of the bands in ``Rrs``
        :return: :py:class:`xarray.Dataset` with ``Rrs`` and ``Rrs_nadir`` (time, wl), the retrieved
            parameters, ``rmsd``
            (root mean square difference with the measured bands, sr-1) and ``success``
        '''
        Rrs = Rrs.transpose('time', band_dim)
        bands = Rrs[band_dim].values.astype(float)
        sza = sza.sel(time=Rrs.time)
        chl = None if chl is None else chl.sel(time=Rrs.time)
        hyper = self._model(self.wl, float(sza[0]))

        ntime = Rrs.sizes['time']
        out = np.full((ntime, self.wl.size), np.nan)
        out_nadir = np.full((ntime, self.wl.size), np.nan)
        pars = np.full((ntime, len(PARAMETERS)), np.nan)
        rmsd = np.full(ntime, np.nan)
        success = np.zeros(ntime, dtype=bool)
        for i in range(ntime):
            values = Rrs.values[i]
            valid = np.isfinite(values)
            sza_ = float(sza[i])
            if valid.sum() < self.min_bands or not np.isfinite(sza_) or sza_ > self.max_sza:
                continue
            model = self._model(bands[valid], sza_)
            model.set_coef(sza_, self.vza, self.azi)
            xinit = dict(XINIT)
            if chl is not None and np.isfinite(chl[i]):
                xinit['chl'] = float(np.clip(chl[i], 0.03, 300))
            measured = np.clip(values[valid], 0, None)
            # sigma only when given: call_solver_scipy of invRrs 1.1.0 has no sigma argument
            weights = {} if self.sigma is None else {'sigma': self.sigma(measured)}
            result = model.call_solver_scipy(measured, xinit=[xinit[name] for name in PARAMETERS],
                                             max_nfev=self.max_nfev, **weights)
            x = np.asarray(result.x, dtype=float)
            fitted = np.asarray(model.forward_model(x, output='Rrs'))
            rmsd[i] = np.sqrt(np.mean((fitted - measured) ** 2))
            success[i] = result.status > 0
            pars[i] = x
            hyper.set_coef(sza_, self.vza, self.azi)
            out[i] = np.asarray(hyper.forward_model(x, output='Rrs'))
            hyper.set_coef(sza_, 0., 90.)
            out_nadir[i] = np.asarray(hyper.forward_model(x, output='Rrs'))
        nfail = int((~success & np.isfinite(rmsd)).sum())
        if nfail:
            logger.warning(f'Rrs reconstruction: fit not converged for {nfail} spectrum(s) '
                           f'(max_nfev={self.max_nfev}), last values kept')

        time = Rrs.time.values
        ds = xr.Dataset({'Rrs': (('time', 'wl'), out), 'Rrs_nadir': (('time', 'wl'), out_nadir)},
                        coords={'time': time, 'wl': self.wl})
        ds['Rrs'].attrs = {'unit': 'sr-1', 'long_name': 'remote sensing reflectance',
                           'description': 'hyperspectral Rrs reconstructed from the multispectral measurements '
                                          'with the invRrs two_modes model',
                           'vza': self.vza, 'azi': self.azi}
        ds['Rrs_nadir'].attrs = {'unit': 'sr-1', 'long_name': 'remote sensing reflectance at nadir',
                                 'description': 'as Rrs, for a nadir view'}
        for k, (name, unit) in enumerate(zip(PARAMETERS, PARAMETER_UNITS)):
            ds[name] = ('time', pars[:, k])
            ds[name].attrs = {'unit': unit, 'description': f'parameter {name} of the invRrs two_modes model'}
        ds['rmsd'] = ('time', rmsd)
        ds['rmsd'].attrs = {'unit': 'sr-1', 'description': 'root mean square difference between the fitted '
                                                           'model and the measured bands'}
        ds['success'] = ('time', success)
        ds.attrs = self.provenance(hyper)
        return ds


def reconstruct_rrs(Rrs, sza, chl=None, band_dim='wl', **kwargs):
    '''
    Hyperspectral reconstruction of multispectral remote-sensing reflectances (see :py:class:`~radcalnet_oc.reconstruction.Reconstruction`,
    which takes ``kwargs``).

    :return: :py:class:`xarray.Dataset` of :py:meth:`Reconstruction.reconstruct`
    '''
    return Reconstruction(**kwargs).reconstruct(Rrs, sza, chl=chl, band_dim=band_dim)
