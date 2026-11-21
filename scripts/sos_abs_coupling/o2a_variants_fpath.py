"""O2-A band f_path = T_atm / T_dir of the v4 variants (ha05, ha1, ha4, aer_ARCT_rh70, polar) against the v4
nominal set.

T_atm = R0_abs / R0_noabs, R0 the diffuse path reflectance (v4: direct glint not simulated), band = Gaussian
764.9 nm FWHM 10 nm (PRISMA-like) and 763.7 nm FWHM 8.8 nm (EnMAP). T_dir: nominal tgdir (same gas profile and
pressure in these variants). Also the water-leaving term T_w (A from albedo 0 / 0.03, S = 0). Medians per band, AOT
and SZA, and differences to the nominal set, printed; all geometries written to v4_o2a/o2a_variants_fpath.csv.

Variants run with make_luts.py, e.g.
    python make_luts.py v4 /data/sos_abs_coupling/v4_o2a 8 --variants=ha05 --wl-range=755,774 --variant-subsample=1 --run
Result (EnMAP 763.7/8.8, 21 November 2026): f_path changes by -0.009/-0.019 (ha05, AOT 0.05/0.2), -0.005/-0.012
(ha1), +0.010/+0.025 (ha4), +0.002/+0.003 (ARCT), -0.004/-0.007 (polar): the aerosol height, the aerosol model and
the polarisation cannot explain the EnMAP O2-A gap (f_path would need to drop by 0.1-0.2, to about 1.0).
"""
import os, sys
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupling as cp

V4 = '/data/sos_abs_coupling/v4'
VAR = '/data/sos_abs_coupling/v4_o2a'
WL = (755., 774.)
BANDS = {'PRISMA': (764.9, 10.), 'EnMAP': (763.7, 8.8)}


def open_set(d):
    out = {}
    for k in ('abs', 'noabs'):
        ds = cp.open_lut(os.path.join(d, f'{k}.nc')).sel(wl=slice(*WL))
        if 'aot550' in ds.dims:
            ds = ds.rename(aot550='aot_ref')
        keep = (ds.gauss_weight > 0) & (ds.vza <= cp.VZA_MAX)
        out[k] = ds.isel(vza=np.flatnonzero(keep.values))
    return out


tg = cp.tg_table(os.path.join(V4, 'nominal')).sel(wl=slice(*WL))
rows = []
for name, d in [('nominal', os.path.join(V4, 'nominal'))] + [(v, os.path.join(VAR, v)) for v in ('ha05', 'ha1', 'ha4', 'aer_ARCT_rh70', 'polar')]:
    if not os.path.exists(os.path.join(d, 'abs.nc')):
        print('missing', d); continue
    lut = open_set(d)
    wl = np.intersect1d(lut['abs'].wl.values, tg.wl.values)
    for k in ('abs', 'noabs'):
        lut[k] = lut[k].sel(wl=wl)
    t = tg.sel(wl=wl)
    R = {k: lut[k].I_up / np.cos(np.radians(lut[k].sza)) for k in ('abs', 'noabs')}
    R0 = {k: R[k].sel(albedo=0) for k in R}
    A = {k: (R[k].sel(albedo=0.03) - R0[k]) / 0.03 for k in R}
    geo = cp.geometry(lut['noabs'])
    Tg_m = cp.tg_of_m(t, geo.m)
    mu0, muv = geo.mu0, geo.muv
    cg = mu0 * muv + np.sqrt(1 - mu0 ** 2) * np.sqrt(1 - muv ** 2) * np.cos(np.radians(lut['noabs'].azi))
    gang = np.degrees(np.arccos(cg.clip(-1, 1)))
    for bname, (c, fw) in BANDS.items():
        g = xr.DataArray(np.exp(-4 * np.log(2) * ((wl - c) / fw) ** 2), coords={'wl': wl})
        band = lambda x: (x * g).sum('wl') / g.sum()
        ds = xr.Dataset({'Tatm': band(R0['abs']) / band(R0['noabs']),
                         'Tdir': band(Tg_m * R0['noabs']) / band(R0['noabs']),
                         'Tw': band(A['abs']) / band(A['noabs']),
                         'Tdir_w': band(Tg_m * A['noabs']) / band(A['noabs']),
                         'R0n': band(R0['noabs']), 'm': geo.m, 'glint_angle': gang})
        df = ds.to_dataframe().reset_index()
        df = df[(df.azi < 360) & (df.vza <= 40) & (df.m < 2.6) & (df.glint_angle > 30)]
        df['f_path'] = df.Tatm / df.Tdir
        df['f_w'] = df.Tw / df.Tdir_w
        df['variant'], df['band'] = name, bname
        rows.append(df)
df = pd.concat(rows)
df = df[df.h2o == df.h2o.min()] if 'h2o' in df else df   # O2 band: H2O irrelevant, avoid triple counting
pd.set_option('display.width', 200)
tab = df.groupby(['band', 'aot_ref', 'sza', 'variant'])[['Tdir', 'Tatm', 'f_path', 'f_w', 'R0n']].median().round(4)
print(tab.to_string())
nom = tab.xs('nominal', level='variant')
print('\nf_path - nominal (median per band, aot):')
print((tab['f_path'] - nom['f_path'].reindex(tab.index.droplevel('variant')).values).unstack('variant').round(4).to_string())
df.to_csv(os.path.join(VAR, 'o2a_variants_fpath.csv'), index=False)
