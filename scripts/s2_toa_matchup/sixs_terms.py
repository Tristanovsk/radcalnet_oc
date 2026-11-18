'''
NIR-SWIR deficit of the Sentinel-2 TOA matchups in view of the 6S formulation of the surface-atmosphere coupling.

6S (Vermote et al., 6S User Guide, Part 1, Eqs. 4-8 and 23-26; Part 2, subroutine ENVIRO) writes the TOA
reflectance as the intrinsic atmospheric reflectance plus the surface contribution: direct-direct, diffuse-direct,
direct-diffuse and multiple interactions (spherical albedo S, trapping 1/(1 - S rho)), and the environment
(adjacency) term T(theta_s) t_d(theta_v) <rho> / (1 - S <rho>) for a non-uniform surface. The OSOAA look-up tables
of radcalnet-oc contain all the coupling terms of the rough sea surface except the direct-direct sunglint (removed
in the SOS core and added back by the matchup script), but they assume an infinite uniform ocean, without foam.
This script quantifies, for the matchups of ``s2_toa_matchup.py`` (selection of ``matchup_analysis.py``):

1. the environment term of the land around each site, with the 6S environment functions of the Rayleigh and
   aerosol scattering, F_R(r) = 1 - [0.930 exp(-0.08 r) + 0.070 exp(-1.10 r)] and
   F_P(r) = 1 - [0.448 exp(-0.27 r) + 0.552 exp(-2.83 r)] (r in km), the Natural Earth 10 m land polygons and a
   vegetated-land reflectance (upper bound);
2. the trapping of the water signal (S rho_w^2) and the whitecap reflectance (Lambertian, absent from OSOAA);
3. the decomposition of the deficit (sim - sat) into an offset and a term proportional to AOT(550), and the
   dependence of each part on the single-scattering geometry factor, the wind, the Angstrom exponent, the glint
   angle and the L1C processing baseline.

Example::

    python sixs_terms.py --odir /data/radcalnet/matchups/sentinel2/v1.2.0
'''
import argparse

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from radcalnet_oc import __version__ as RADCALNET_OC_VERSION  # noqa: E402
ODIR = f'/data/radcalnet/matchups/sentinel2/v{RADCALNET_OC_VERSION}'   # one folder per radcalnet_oc version
SITES = {'AAOT_lev20': (45.3139, 12.5083), 'Galata_Platform_lev20': (43.044624, 28.19319),
         'AAOT_lev15': (45.3139, 12.5083), 'Section-7_Platform_lev15': (44.5458, 29.4466)}
POOLED = ['AAOT_lev20', 'Galata_Platform_lev20', 'AAOT_lev15']
BANDS = [865, 1610, 2190]
#: reflectance of vegetated land (upper bound of the environment reflectance)
RHO_LAND = {865: 0.30, 1610: 0.22, 2190: 0.12}
#: spectral factor of the whitecap reflectance relative to the visible (Frouin et al., 1996)
FOAM_FACTOR = {865: 0.75, 1610: 0.35, 2190: 0.20}


# ---------------------------------------------------------------------------------------------------
# 6S terms
# ---------------------------------------------------------------------------------------------------
def env_density(r, kind):
    '''dF/dr of the 6S environment function (per km of radius), Rayleigh ``R`` or aerosol ``P``.'''
    if kind == 'R':
        return 0.930 * 0.08 * np.exp(-0.08 * r) + 0.070 * 1.10 * np.exp(-1.10 * r)
    return 0.448 * 0.27 * np.exp(-0.27 * r) + 0.552 * 2.83 * np.exp(-2.83 * r)


def environment_weights(lat, lon, land, rmax=80., step=0.25):
    '''
    Fraction of the 6S environment weighting falling on land around (lat, lon), for the Rayleigh (W_R) and
    aerosol (W_P) functions, and distance to the coast (km).
    '''
    to_km = Transformer.from_crs(4326, f'+proj=aeqd +lat_0={lat} +lon_0={lon} +units=km', always_xy=True)
    local = shp_transform(lambda x, y: to_km.transform(x, y),
                          land.intersection(shapely.box(lon - 1.2, lat - 0.9, lon + 1.2, lat + 0.9)))
    x = np.arange(-rmax, rmax + step, step)
    X, Y = np.meshgrid(x, x)
    R = np.maximum(np.hypot(X, Y), step / 2)
    on_land = shapely.contains_xy(local, X, Y) & (R <= rmax)
    out = {'coast_km': float(local.distance(shapely.Point(0, 0)))}
    for kind in 'RP':
        out[f'W_{kind}'] = float((env_density(R, kind) / (2 * np.pi * R) * on_land).sum() * step ** 2)
    return out


def rayleigh_tau(wl):
    '''Rayleigh optical thickness at sea level (wl in nm).'''
    w = wl / 1e3
    return 0.008569 * w ** -4 * (1 + 0.0113 * w ** -2 + 0.00013 * w ** -4)


def diffuse_rayleigh(tau, mu):
    '''Diffuse transmittance of the Rayleigh scattering (6S, delta-Eddington, Eq. 15 of Part 1).'''
    return ((2 / 3 + mu) + (2 / 3 - mu) * np.exp(-tau / mu)) / (4 / 3 + tau) - np.exp(-tau / mu)


def diffuse_aerosol(tau, mu, forward=0.85, ssa=0.95):
    '''Diffuse transmittance of the aerosols (fraction ``forward`` of the scattering kept in the beam).'''
    return np.exp(-tau / mu) * (np.exp(ssa * forward * tau / mu) - 1)


def environment_term(weights, sza, vza, aot550, ang_exp, wl):
    '''6S environment term T(theta_s) [t_dR W_R + t_dP W_P] rho_land (reflectance) for the land of the site.'''
    mus, muv = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    tau_r, tau_a = rayleigh_tau(wl), aot550 * (wl / 550) ** -ang_exp
    T_s = np.exp(-(tau_r + tau_a) / mus) + diffuse_rayleigh(tau_r, mus) + diffuse_aerosol(tau_a, mus)
    return T_s * (diffuse_rayleigh(tau_r, muv) * weights['W_R'] + diffuse_aerosol(tau_a, muv) * weights['W_P']) \
        * RHO_LAND[wl]


# ---------------------------------------------------------------------------------------------------
# matchups
# ---------------------------------------------------------------------------------------------------
def load(odir, site, dt_max=60, aot_max=0.3, glint_min=30):
    '''Selected matchups (selection of matchup_analysis.py), deficit sim - sat in 1e-3.'''
    ds = xr.open_dataset(f'{odir}/{site}_s2_toa_matchup.nc')
    sel = ((ds.frac_valid > 0.5) & (ds.n_pix >= 50) & (np.abs(ds.dt_nearest) <= dt_max)
           & (ds.aot550 < aot_max) & (ds.glint_angle > glint_min))
    ds = ds.sel(time=sel)
    df = pd.DataFrame({f'D{b}': 1e3 * (ds.sim_median - ds.sat_median).sel(wl=b).values for b in BANDS},
                      index=ds.time.values)
    for v in ['aot550', 'ang_exp', 'wind', 'sza', 'vza', 'glint_angle']:
        df[v] = ds[v].values
    df['baseline'] = [str(p).split('_')[3] for p in ds['product'].values]
    df['G'] = 1 / (4 * np.cos(np.radians(df.sza)) * np.cos(np.radians(df.vza)))
    return df.dropna(subset=['D865', 'aot550'])


def ols(df, cols, y):
    '''Least squares y ~ 1 + cols: coefficients, t values, R2, n.'''
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    Y = df[y].values
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    se = np.sqrt(np.diag(res @ res / (len(Y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return beta, beta / se, 1 - res @ res / ((Y - Y.mean()) @ (Y - Y.mean())), len(Y)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--odir', default=ODIR, help='directory of the matchup files')
    args = p.parse_args()
    import cartopy.io.shapereader as shpreader
    land = shapely.unary_union(list(shpreader.Reader(
        shpreader.natural_earth('10m', 'physical', 'land')).geometries()))
    pd.set_option('display.width', 200)

    frames = {site: load(args.odir, site) for site in SITES}

    print('=== 1. 6S environment term (median geometry and AOT of the site, vegetated land), x1e-3')
    rows = {}
    for site, (lat, lon) in SITES.items():
        df, w = frames[site], environment_weights(lat, lon, land)
        r = {'N': len(df), 'coast_km': w['coast_km'], 'W_R': w['W_R'], 'W_P': w['W_P']}
        for b in BANDS:
            r[f'env{b}'] = 1e3 * environment_term(w, df.sza.median(), df.vza.median(), df.aot550.median(),
                                                  df.ang_exp.median(), b)
            r[f'D{b}'] = df[f'D{b}'].median()
        rows[site] = r
    print(pd.DataFrame(rows).T.round(4).to_string())

    print('\n=== 2. trapping of the water signal T T S rho_w^2 / (1 - S rho_w), x1e-3')
    for rho_w in [0.002, 0.01, 0.05]:
        print(f'  rho_w = {rho_w}: ' + ', '.join(f'S = {S}: {1e3 * S * rho_w ** 2 / (1 - S * rho_w):.5f}'
                                                 for S in [0.05, 0.15]))
    print('    whitecaps T T W rho_foam, W = 2.95e-6 U^3.52, rho_foam = 0.22 x spectral factor, x1e-3')
    for U in [4, 6, 8, 10]:
        W = 2.95e-6 * U ** 3.52
        print(f'  U = {U:2d} m/s: ' + ', '.join(f'{b}: {1e3 * 0.9 * 0.22 * W * FOAM_FACTOR[b]:.3f}' for b in BANDS))

    pool = pd.concat([frames[s] for s in POOLED])
    pool['aotG'] = pool.aot550 * pool.G
    pool['wind_foam'] = 1e-3 * pool.wind ** 3.52
    print(f'\n=== 3. deficit (x1e-3) = offset + slope x AOT(550), sites {", ".join(POOLED)} (n = {len(pool)})')
    slopes = {}
    for cols in (['aot550'], ['aot550', 'wind_foam'], ['aot550', 'ang_exp'], ['aotG']):
        for b in BANDS:
            beta, t, r2, n = ols(pool, cols, f'D{b}')
            if cols == ['aot550']:
                slopes[b] = beta[1]
            print(f'  D{b} ~ 1 + {" + ".join(cols):18s} R2 = {r2:.2f}  ' +
                  '  '.join(f'{c} = {v:+.2f} (t = {tt:+.1f})' for c, v, tt in zip(['1'] + cols, beta, t)))
    for site, df in frames.items():
        print(f'  {site}: ' + '  '.join('D{} = {:+.2f} {:+.2f} x AOT'.format(b, *ols(df, ['aot550'], f'D{b}')[0])
                                        for b in BANDS))

    print('\n    median deficit per wind bin (AOT < 0.15)')
    low = pool[pool.aot550 < 0.15]
    print(low.groupby(pd.cut(low.wind, [0, 3, 5, 7, 15]), observed=True)[[f'D{b}' for b in BANDS]]
          .median().assign(N=low.groupby(pd.cut(low.wind, [0, 3, 5, 7, 15]), observed=True).size()).round(2).to_string())
    print('\n    median deficit per Angstrom exponent bin')
    g = pool.groupby(pd.cut(pool.ang_exp, [0, 0.8, 1.2, 1.6, 3]), observed=True)
    print(g[['D865', 'D2190', 'aot550']].median().assign(N=g.size()).round(3).to_string())
    print('\n    AOT-independent part (D - slope x AOT) per glint-angle bin')
    for b in BANDS:
        pool[f'off{b}'] = pool[f'D{b}'] - slopes[b] * pool.aot550
    g = pool.groupby(pd.cut(pool.glint_angle, [30, 40, 50, 60, 90]), observed=True)
    print(g[[f'off{b}' for b in BANDS] + ['sza']].median().assign(N=g.size()).round(2).to_string())
    print('\n    AAOT L2.0: deficit at 865 nm per L1C processing baseline')
    d = frames['AAOT_lev20']
    print(d.groupby('baseline').agg(N=('D865', 'size'), D865=('D865', 'median')).round(2).to_string())


if __name__ == '__main__':
    main()
