'''
AERONET almucantar phase function at AAOT against the radcalnet_oc OPAC mixture, at 865 nm.

AERONET V3 inversions (Level 1.5 with the Level 2 quality flag, without the AOD(440) > 0.4 threshold; files of
``download_aeronet_inversions.py``). For each retrieval, the mixture of radcalnet_oc (``kernel.Aerosol``:
DESE_rh70, MACL_rh70, WASO_rh0, proportions of the AOT at 550 nm, bounded least squares on the normalized AOT) is
refitted on the AERONET extinction AOD at 443, 667, 865 and 1020 nm. The LUT tables of the models are combined
linearly, so the effective omega*P of the mixture at 865 nm is sum c_i naot_i w_i P_i / sum c_i naot_i, with the
mopsmap OPAC matrices of the LUT (Cext and Csca for naot and the single scattering albedo, linear interpolation
between the 50 nm files). Both phase functions are normalised to (1/2) int P sin(theta) dtheta = 1.

Results of 27 November 2026 (4670 retrievals, 2018-2026; mixture median 95 % WASO_rh0, tau865 of the mixture 0.99
of the measured one): AERONET omega*P(865) is 12-22 % above the mixture at all scattering angles from 90 to 180 deg,
+0.03 at 90-150 deg and up to +0.05 at 160-170 deg. No single OPAC component closes the gap: humid WASO and the coarse
models (MACL, DESE) have less backscatter at 865 nm than WASO_rh0.

Caveats: the almucantar measures scattering angles up to 2 sza only (median 124 deg at AAOT), beyond which the
phase function is the extrapolation of the retrieval (column d_wP_measured_only: the difference over the measured
range only); the single scattering albedo is unreliable at low AOD (it contributes about 2 of the 13 %).

Writes ``<AEROSOL_PHASE_DIR>/pfn_compare_AAOT.npz`` for ``pfn_predict.py``.

Usage:
    python pfn_compare.py
'''
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from phase_common import MOPSMAP, STUDY_DIR, read_aeronet

warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)

TH = np.arange(90, 180.1, 1.0)
WLS = [443, 667, 865, 1020]
FIT_MODELS = ['DESE_rh70', 'MACL_rh70', 'WASO_rh0']
OTHER_MODELS = ['WASO_rh50', 'WASO_rh70', 'COAV_rh70', 'URBA_rh70', 'ARCT_rh70', 'MACL_rh90', 'MACL_rh99']
BINS = [(90, 120), (120, 130), (130, 140), (140, 150), (150, 160), (160, 170), (170, 180)]
OUT = os.path.join(STUDY_DIR, 'pfn_compare_AAOT.npz')


def mopsmap(model, wl):
    '''Cext, single scattering albedo and F11 on TH at wl (nm), between the 50 nm mopsmap files.'''
    lo, hi = 50 * np.floor(wl / 50), 50 * np.ceil(wl / 50)
    out = []
    for w in (lo, hi):
        f = MOPSMAP.format(model=model, wl=w)
        with open(f) as fh:
            cext = float(fh.readline().split(':')[1]); csca = float(fh.readline().split(':')[1])
        M = np.loadtxt(f, skiprows=4)
        out.append((cext, csca / cext, np.interp(TH, M[:, 0], M[:, 1])))
    x = 0. if hi == lo else (wl - lo) / (hi - lo)
    cext = np.exp((1 - x) * np.log(out[0][0]) + x * np.log(out[1][0]))
    return cext, (1 - x) * out[0][1] + x * out[1][1], (1 - x) * out[0][2] + x * out[1][2]


def optics(model):
    c550 = mopsmap(model, 550)[0]
    _, ssa, P = mopsmap(model, 865)
    return dict(naot=np.array([mopsmap(model, w)[0] / c550 for w in WLS]), ssa=ssa, P=P)


if __name__ == '__main__':
    pfn = read_aeronet('AAOT_PFN_ALM15.txt')
    pfn = pfn[pfn['If_Retrieval_is_L2(without_L2_0.4_AOD_440_threshold)'] == 1]
    aod, ssa = read_aeronet('AAOT_AOD_ALM15.txt'), read_aeronet('AAOT_SSA_ALM15.txt')
    cols = [c for c in pfn.columns if c.endswith('[865nm]') and c[0].isdigit()]
    ang = np.array([float(c.split('[')[0]) for c in cols]); o = np.argsort(ang)
    P_A = np.array([np.interp(TH, ang[o], r[o]) for r in pfn[cols].values])
    tau = aod.reindex(pfn.index)[[f'AOD_Extinction-Total[{w}nm]' for w in WLS]].values
    w865 = ssa.reindex(pfn.index)['Single_Scattering_Albedo[865nm]'].values
    sza = pfn['Solar_Zenith_Angle_for_Measurement_Start(Degrees)'].values
    print(f'retrievals {len(pfn)} ({pfn.index.min():%Y-%m} - {pfn.index.max():%Y-%m}); SSA865 5/50/95 %: '
          f'{np.nanpercentile(w865, [5, 50, 95]).round(3)}; largest measured scattering angle (2 sza), median {np.median(2 * sza):.0f} deg')

    opt = {m: optics(m) for m in FIT_MODELS + OTHER_MODELS}
    for m, v in opt.items():
        print(f'{m:10s} naot(443,667,865,1020) {np.round(v["naot"], 3)} ssa865 {v["ssa"]:.3f} '
              f'P865(120,140,160,175) {np.round(np.interp([120, 140, 160, 175], TH, v["P"]), 3)}')

    lt = np.log(tau)
    t550 = np.exp(lt[:, 0] + (lt[:, 1] - lt[:, 0]) * np.log(550 / 443) / np.log(667 / 443))
    naot = tau / t550[:, None]
    N3 = np.array([opt[m]['naot'] for m in FIT_MODELS])
    ok = np.isfinite(naot).all(1) & np.isfinite(w865)
    F = np.full((len(pfn), 3), np.nan)
    for i in np.where(ok)[0]:
        r = least_squares(lambda f: naot[i] - (f / f.sum()) @ N3, [0, .5, .5], bounds=([0] * 3, [1] * 3))
        F[i] = r.x / r.x.sum()
    print('\nfitted proportions (median):', dict(zip(FIT_MODELS, np.nanmedian(F, 0).round(3))),
          '| retrievals with WASO_rh0 > 0.9:', f'{np.nanmean(F[:, 2] > .9):.0%}')
    k = WLS.index(865)
    den = sum(F[:, [i]] * opt[m]['naot'][k] for i, m in enumerate(FIT_MODELS))
    wP_M = sum(F[:, [i]] * opt[m]['naot'][k] * opt[m]['ssa'] * opt[m]['P'][None, :] for i, m in enumerate(FIT_MODELS)) / den
    wP_A = w865[:, None] * P_A
    print('tau865 mixture / measured, median:', np.nanmedian(den[:, 0] * t550 / tau[:, k]).round(3))

    dwP = (wP_A - wP_M)[ok]
    rows = []
    for a, b in BINS:
        j = (TH > a) & (TH <= b)
        meas = 2 * sza[ok] >= b
        v, va, vm = dwP[:, j].mean(1), wP_A[ok][:, j].mean(1), wP_M[ok][:, j].mean(1)
        rows.append({'theta': f'({a},{b}]', 'wP_aeronet': np.median(va), 'wP_mixture': np.median(vm), 'd_wP': np.median(v),
                     'q25': np.percentile(v, 25), 'q75': np.percentile(v, 75), 'rel_pct': 100 * np.median(v / vm),
                     'd_wP_measured_only': np.median(v[meas]) if meas.sum() > 20 else np.nan, 'n_measured': int(meas.sum())})
    print('\nomega*P at 865 nm, AERONET - radcalnet_oc mixture of the same retrieval')
    print(pd.DataFrame(rows).set_index('theta').round(3).to_string())
    print('\nAERONET - single OPAC model, omega*P at 865 nm (median per bin)')
    alt = {m: [np.median(wP_A[ok][:, (TH > a) & (TH <= b)].mean(1)) - opt[m]['ssa'] * opt[m]['P'][(TH > a) & (TH <= b)].mean()
               for a, b in BINS] for m in opt}
    print(pd.DataFrame(alt, index=[f'({a},{b}]' for a, b in BINS]).T.round(3).to_string())

    os.makedirs(STUDY_DIR, exist_ok=True)
    np.savez(OUT, TH=TH, dwP=dwP, time=pfn.index[ok].values.astype('datetime64[s]').astype('int64'), sza=sza[ok])
    print('\nwritten', OUT)
