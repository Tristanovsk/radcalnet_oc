'''
Inputs shared by the scripts of the NIR excess and aerosol phase function study (ROADMAP task 10).

Matchup files of radcalnet_oc 1.2.0 (Sentinel-3 OLCI Cal/Val monitoring, Sentinel-2 MSI matchups with the heavy-tailed
glint in the region of interest), the AERONET inversions of AAOT, and the mopsmap OPAC scattering matrices of the TOA
LUT. The paths can be changed with the environment variables below.
'''
import os

import numpy as np
import pandas as pd
import xarray as xr

S3_MATCHUP = os.environ.get(
    'S3_MATCHUP',
    '/data/radcalnet/matchups/sentinel3/v1.2.0/olci_radcalnet-oc_v1.2.0_VEIT-GABU-CORO_20160401-20261231_matchup.nc')
S2_DIR = os.environ.get('S2_MATCHUP_DIR', '/data/radcalnet/matchups/sentinel2/v1.2.0')
S2_FILES = {'VEIT': 'AAOT_lev20_s2_toa_matchup_cm_iso_mix_roi.nc',
            'CORO': 'Section-7_Platform_lev15_s2_toa_matchup_cm_iso_mix_roi.nc'}
STUDY_DIR = os.environ.get('AEROSOL_PHASE_DIR', '/data/radcalnet/studies/aerosol_phase')
AERONET_DIR = os.path.join(STUDY_DIR, 'aeronet')
MOPSMAP = os.environ.get('MOPSMAP', '/DATA/git/vrtc/mopsmap_tbx/scatmat/ascii/scat_mat_opac_{model}_wl{wl:.1f}.csv')


def geometry(sza, vza, azi):
    '''Scattering angle (deg) and single-scattering air-mass factor 1 / (4 mu_s mu_v).

    The relative azimuth of the matchup files is 0 in the specular direction: the glint angle is
    arccos(cos sza cos vza + sin sza sin vza cos azi), which these files reproduce exactly.'''
    s, v, a = np.radians(sza), np.radians(vza), np.radians(azi)
    theta = np.degrees(np.arccos(np.sin(s) * np.sin(v) * np.cos(a) - np.cos(s) * np.cos(v)))
    return theta, 1 / (4 * np.cos(s) * np.cos(v))


def tau865(aot550, ang):
    '''AOT at 865 nm from AOT(550) and the Angstrom exponent.'''
    return aot550 * (865 / 550) ** (-ang)


def load_s3(site=None):
    '''OLCI matchups without high or near-specular glint (selection of the site statements).

    Returns the dataset and the excess OLCI - simulation in TOA reflectance (time, band).'''
    ds = xr.open_dataset(S3_MATCHUP)
    keep = ~pd.Series(ds.glint_level.values).isin(['high', 'near_specular']).values
    if site is not None:
        keep &= ds.site.values == site
    d = ds.isel(time=keep)
    return d, d.Rsim * (d.ratio_refl - 1)


def load_s2(site):
    '''MSI matchups: valid fraction > 0.5, at least 50 pixels, high and near-specular glint flags excluded,
    AOT(550) < 0.3, nearest AERONET-OC measurement within 60 min.

    Returns the dataset and the excess MSI - simulation with the direct glint (time, wl).'''
    ds = xr.open_dataset(os.path.join(S2_DIR, S2_FILES[site]))
    flags = ds.glint_flags.fillna(0).astype(int).values
    keep = ((ds.frac_valid.values > 0.5) & (ds.n_pix.values >= 50) & ((flags & (4 | 8)) == 0)
            & (ds.aot550.values < 0.3) & (np.abs(ds.dt_nearest.values) <= 60))
    d = ds.isel(time=keep)
    return d, d.sat_median - d.simg_median


def read_aeronet(name):
    '''AERONET V3 inversion file (all points) with a datetime index.'''
    d = pd.read_csv(os.path.join(AERONET_DIR, name), skiprows=6, na_values=[-999, -999.0])
    d.index = pd.to_datetime(d['Date(dd:mm:yyyy)'] + ' ' + d['Time(hh:mm:ss)'], format='%d:%m:%Y %H:%M:%S')
    return d
