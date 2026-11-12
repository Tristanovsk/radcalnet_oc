"""Band-level comparison of the PPLD model with the adopted parameterization (eta / blend) and the decoupled
model, on the coupled SOS-ABS simulations (v2: fit, v1: independent test).

Usage: python ppld_eval.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupling as cp
import ppld
from radcalnet_oc import SolarIrradiance

SRF = os.path.join(HERE, '..', '..', 'notebook', 'flex', 'S3A_OL_SRF_20160713_mean_rsr.nc4')


def gauss(c, fwhm):
    x = np.linspace(c - 1.5 * fwhm, c + 1.5 * fwhm, 301)
    return x, np.exp(-4 * np.log(2) * ((x - c) / fwhm) ** 2)


def bands():
    olci = xr.open_dataset(SRF)
    ob = lambda n: (olci.mean_spectral_response_function_wavelength.values[n - 1],
                    olci.mean_spectral_response_function.values[n - 1])
    return {'O2-B 687': gauss(687.5, 5), 'H2O 722': gauss(722, 10), 'Oa13': ob(13), 'Oa14': ob(14), 'Oa15': ob(15),
            'H2O 822': gauss(822, 10), 'Oa19': ob(19), 'Oa20': ob(20), 'MSI B9': gauss(945, 20)}


def band_weights(wl, B, F0):
    e0 = F0.interp(wl=wl).values * np.any([(wl >= a) & (wl <= b) for a, b in cp.RANGES], axis=0)
    W = np.array([np.interp(wl, x, s, left=0, right=0) * e0 for x, s in B.values()])
    return xr.DataArray(W / W.sum(1, keepdims=True), coords={'band': list(B), 'wl': wl}, dims=('band', 'wl'))


def band_error(eff, Ta, Tw, W, rho=0.005):
    mod, tru = cp.toa(eff, Ta, Tw, rho)
    return (xr.dot(mod.fillna(0), W, dim='wl') / xr.dot(tru, W, dim='wl') - 1).where(eff['geo'].valid) * 100


def main():
    B = bands()
    F0 = SolarIrradiance().tsis
    eff = {s: cp.effective(cp.load(d)) for s, d in (('v2', cp.DATA_V2), ('v1', cp.DATA_V1))}
    adopted = cp.from_table()
    t = time.time()
    a = {s: ppld.exponential_sum(e['tg'])[0] for s, e in eff.items()}
    model = ppld.fit_all(eff['v2'], a['v2'])
    print(f'PPLD fit: {time.time() - t:.0f} s')
    rows = []
    for s, e in eff.items():
        W = band_weights(e['T'].wl.values, B, F0)
        cases = {'decoupled': (e['T'].Tg_m, e['T'].Tg_m),
                 'adopted': (cp.apply(e, adopted, 'atm'), cp.apply(e, adopted, 'w')),
                 'ppld': (ppld.apply(e, model, a[s], 'atm'), ppld.apply(e, model, a[s], 'w'))}
        for name, (Ta, Tw) in cases.items():
            err = band_error(e, Ta, Tw, W)
            for b in B:
                x = abs(err.sel(band=b))
                rows.append({'set': s, 'model': name, 'band': b, 'median': float(x.median()),
                             'p95': float(x.quantile(0.95))})
        # irradiance and water-leaving term: median |T_model / T_sim - 1| at band level
        for name, (te, tw) in {'decoupled': (e['T'].Tg_m0, e['T'].Tg_m),
                               'adopted': (cp.apply(e, adopted, 'Ed'), cp.apply(e, adopted, 'w')),
                               'ppld': (ppld.apply(e, model, a[s], 'Ed'), ppld.apply(e, model, a[s], 'w'))}.items():
            ed = (xr.dot(te * e['T'].Ed_noabs, W, dim='wl') / xr.dot(e['T'].Ed_abs, W, dim='wl') - 1) * 100
            for b in B:
                rows.append({'set': s, 'model': name, 'band': b, 'Ed median': float(abs(ed.sel(band=b)).median())})
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    for s in ('v2', 'v1'):
        d = df[(df.set == s) & df['median'].notna()]
        print(f'\n{s}: |band error of R_toa| (%), median [p95], rho = 0.005')
        t = d.assign(cell=d['median'].round(2).astype(str) + ' [' + d['p95'].round(1).astype(str) + ']')
        print(t.pivot(index='band', columns='model', values='cell').reindex(list(B))[['decoupled', 'adopted', 'ppld']].to_string())
        d = df[(df.set == s) & df['Ed median'].notna()]
        print(f'{s}: |band error of Ed| (%), median')
        print(d.pivot(index='band', columns='model', values='Ed median').reindex(list(B))[['decoupled', 'adopted', 'ppld']].round(3).to_string())
    return model, a, eff, df


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


# ---------------------------------------------------------------------------------------------------------------
# transfer to the absorption tables of the package (as in the analysis notebook, section 6)
def transfer(e, o2_model, coupling, term, B, F0, gas_lut, end_to_end=False):
    from radcalnet_oc import GaseousTransmittance
    T, dec, geo = e['T'], e['dec'], e['geo']
    if term == 'Ed':
        m, q_abs, base, tg_ckd, valid = geo.m0, T.Ed_abs, T.Ed_noabs, T.Tg_m0, True
    else:
        q = 'R0' if term == 'atm' else 'A'
        az = dict(azi=slice(0, None, 6))
        m, tg_ckd = geo.m, T.Tg_m.isel(**az)
        q_abs, base, valid = dec['abs'][q].isel(**az), dec['noabs'][q].isel(**az), geo.valid.isel(**az)
    gt = GaseousTransmittance(gas_lut, o2_model=o2_model)
    gt.pressure = 1013.
    gt.gas_tc['h2o'] = T.h2o * 10.
    gt.air_mass = m
    t0 = gt.get_gaseous_transmittance()
    tc = t0 if coupling is None else gt.get_coupled_transmittance(m, term, T.aot_ref, coupling)
    wsos, wpkg = band_weights(T.wl.values, B, F0), band_weights(t0.wl.values, B, F0)
    b = base.interp(wl=t0.wl.values, kwargs={'fill_value': 'extrapolate'}).fillna(0)
    if end_to_end:
        norm = xr.dot(base.fillna(0), wsos, dim='wl') / xr.dot(b, wpkg, dim='wl')
        err = xr.dot(tc, b, wpkg, dim='wl') * norm / xr.dot(q_abs, wsos, dim='wl') - 1
    else:
        c_sim = xr.dot(q_abs, wsos, dim='wl') / xr.dot((tg_ckd * base).fillna(0), wsos, dim='wl')
        err = xr.dot(tc, b, wpkg, dim='wl') / xr.dot(t0, b, wpkg, dim='wl') / c_sim - 1
    return (err * 100).where(valid)


def transfer_main():
    from radcalnet_oc import LUT, GasCoupling
    from radcalnet_oc.kernel import GAS_COUPLING_FILE, GAS_COUPLING_PPLD_FILE
    B = bands()
    F0 = SolarIrradiance().tsis
    e = cp.effective(cp.load(cp.DATA_V2))
    gas_lut = LUT().gas_lut.sel(wl=slice(680, 995))
    models = {'decoupled': None, 'effective_air_mass': GasCoupling(GAS_COUPLING_FILE),
              'path_length': GasCoupling(GAS_COUPLING_PPLD_FILE)}
    rows = []
    for term in cp.TERMS:
        for o2 in ('lbl', 'reptran'):
            for name, c in models.items():
                t = time.time()
                x = transfer(e, o2, c, term, B, F0, gas_lut)
                dt = time.time() - t
                for b in B:
                    v = abs(x.sel(band=b))
                    rows.append({'term': term, 'o2': o2, 'model': name, 'band': b, 'median': float(v.median()),
                                 'p95': float(v.quantile(0.95)), 'time': dt})
                if term == 'atm':
                    x = transfer(e, o2, c, term, B, F0, gas_lut, end_to_end=True)
                    for b in B:
                        v = x.sel(band=b)
                        rows.append({'term': 'atm end-to-end', 'o2': o2, 'model': name, 'band': b,
                                     'median': float(v.median()), 'p95': float(v.quantile(0.95)), 'time': dt})
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for term in ('atm', 'w', 'Ed', 'atm end-to-end'):
        d = df[df.term == term]
        lab = 'signed median [p95]' if 'end' in term else '|C_pkg/C_sim - 1| median [p95]'
        print(f'\n{term}: {lab} (%)')
        cell = d['median'].round(1).astype(str) + ' [' + d['p95'].round(1).astype(str) + ']'
        print(d.assign(cell=cell).pivot_table(index='band', columns=['o2', 'model'], values='cell',
                                              aggfunc='first').reindex(list(B)).to_string())
    print('\ncomputing time of the transmittances (s, atm, v2 geometries):')
    print(df[df.term == 'atm'].groupby(['o2', 'model'])['time'].first().round(2).to_string())
    return df


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'transfer':
    transfer_main()
