'''Usage: python grs_pressure_stats.py <output folder of grs_pressure_check.py (results/ inside)>

Statistics of the GRS pressure check: 'linear' vs 'rayleigh' against AERONET-OC, and change with the pressure.'''
import glob
import os
import sys

import numpy as np
import pandas as pd

S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
M = '/data/satellite/Sentinel-2/test/aerosol_settings_aeronet/matchups.csv'
WL = [443, 490, 560, 665]

res = pd.concat([pd.read_csv(f) for f in glob.glob(f'{S}/results/*.csv')], ignore_index=True)
m = pd.read_csv(M).drop_duplicates('path')
m['image'] = m.path.map(lambda p: os.path.basename(p).replace('.SAFE', ''))
df = res.merge(m[['image', 'site', 'time', 'is_aot550', *[f'is_rrs{w}' for w in WL]]], on='image')
ok = df.station_in_image & (df.box_mask_valid >= 5) & df.aot550.notnull()
valid = df[ok].groupby('image').filter(lambda g: len(g) == 2)
print(f'{df.image.nunique()} images processed, {valid.image.nunique()} valid (station in image, >= 5 valid pixels)')

lin = valid[valid.pressure_scaling == 'linear'].set_index('image')
ray = valid[valid.pressure_scaling == 'rayleigh'].set_index('image').loc[lin.index]
P = lin.pressure / 100
print(f'pressure: median {P.median():.1f} hPa, range {P.min():.1f}-{P.max():.1f}')

# change rayleigh - linear vs pressure (regression on P - 1015.2)
rows = []
for v in ['aot550', *[f'rrs{w}' for w in [443, 490, 560, 665, 865]]]:
    d = ray[v] - lin[v]
    a, b = np.polyfit(P - 1015.2, d, 1) if P.nunique() > 2 else (np.nan, np.nan)
    rows.append(dict(var=v, median_delta=d.median(), p5=d.quantile(.05), p95=d.quantile(.95),
                     slope_per_10hPa=10 * a, at_1015=b, median_rel=(d / lin[v].abs()).median()))
print('\nrayleigh - linear (slope against P - 1015.2 hPa):')
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.2e}'))

# agreement with AERONET-OC
out = []
for site, idx in [('all', lin.index), *[(s, lin.index[lin.site == s]) for s in lin.site.unique()]]:
    for v in ['aot550', *[f'rrs{w}' for w in WL]]:
        ref = lin.loc[idx, f'is_{v}']
        for name, t in (('linear', lin), ('rayleigh', ray)):
            e = (t.loc[idx, v] - ref).dropna()
            if len(e) < 3:
                continue
            out.append(dict(site=site, var=v, scaling=name, n=len(e), bias=e.mean(), median=e.median(),
                            rmsd=np.sqrt((e ** 2).mean()), mad=e.abs().median(),
                            mapd=(e / ref.loc[e.index]).abs().median() * 100))
stats = pd.DataFrame(out)
stats.to_csv(f'{S}/stats.csv', index=False)
print('\nagainst AERONET-OC:')
print(stats.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
