'''
Does the AERONET - OPAC difference of omega*P(865) predict the NIR excess of the VEIT matchups?

Single-scattering prediction of the missing aerosol path reflectance at 865 nm,

    dR_pred = tau865 / (4 mu_s mu_v) * d_wP(theta),

with d_wP the median of the same-day AAOT inversions (``pfn_compare.py``), else of the month, at the scattering
angle of the scene; tau865 from the AERONET-OC AOT(550) and Angstrom exponent of the matchup. The observed excess
is regressed on dR_pred, and on a control with the same tau / (4 mu_s mu_v) scaling but a constant d_wP (no angular
shape, no day-to-day change).

Results of 27 November 2026: OLCI Oa17 (N 737): mean excess 2.40e-3, predicted 1.32e-3, slope 0.59 +/- 0.08,
intercept 1.62e-3; MSI B8A (N 325): 2.33e-3, predicted 0.92e-3, slope 0.80 +/- 0.19, intercept 1.59e-3. The excess
increases monotonically over the quintiles of the prediction (OLCI 1.05 to 3.24e-3), the control has no skill
(R2 0.003 against 0.05-0.075), and in the joint regression only the AERONET term is significant (t 7.6 and 4.2).
About 1e-3 remains where the prediction is zero.

Usage:
    python pfn_predict.py
'''
import os
import warnings

import numpy as np
import pandas as pd

from phase_common import STUDY_DIR, geometry, load_s2, load_s3, tau865

warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)

z = np.load(os.path.join(STUDY_DIR, 'pfn_compare_AAOT.npz'))
TH, DWP = z['TH'], z['dwP']
T = pd.to_datetime(z['time'], unit='s')
DAILY = {d: np.median(DWP[T.normalize() == d], 0) for d in np.unique(T.normalize())}
MONTHLY = {m: np.median(DWP[T.month == m], 0) for m in range(1, 13)}
ALL = np.median(DWP, 0)


def predict(times, theta, x):
    out, same_day = [], []
    for t, a, xi in zip(pd.to_datetime(times), theta, x):
        c = DAILY.get(t.normalize())
        same_day.append(c is not None)
        if c is None:
            c = MONTHLY.get(t.month, ALL)
        out.append(xi * np.interp(a, TH, c, left=np.nan))
    return np.array(out), np.array(same_day)


def fit(X, y):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    b, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    r = y[ok] - A @ b
    se = np.sqrt(np.diag(r @ r / (ok.sum() - A.shape[1]) * np.linalg.inv(A.T @ A)))
    return b, se, 1 - r.var() / y[ok].var(), ok


def test(name, d, dR):
    theta, m = geometry(d.sza.values, d.vza.values, d.azi.values)
    x = m * tau865(d.aot550.values, d.ang_exp.values)
    pr, same = predict(d.time.values, theta, x)
    b, se, r2, ok = fit(pr[:, None], dR)
    print(f'\n{name}: N {ok.sum()}, same-day inversion {same[ok].mean():.0%}')
    print(f'  mean excess {1e3 * dR[ok].mean():.2f}e-3, mean predicted {1e3 * pr[ok].mean():.2f}e-3')
    print(f'  dR = {1e3 * b[0]:.2f}e-3 (+/-{1e3 * se[0]:.2f}) + {b[1]:.2f} (+/-{se[1]:.2f}) dR_pred, '
          f'r {np.corrcoef(pr[ok], dR[ok])[0, 1]:.2f}, R2 {r2:.3f}')
    q = pd.qcut(pr[ok], 5)
    t = pd.DataFrame({'pred_1e3': 1e3 * pr[ok], 'obs_1e3': 1e3 * dR[ok]}).groupby(q, observed=True)
    print('  by quintile of the prediction:'); print(t.median().round(2).assign(N=t.size()).to_string())
    pc = x * np.nanmedian(ALL)
    bc, sec, r2c, _ = fit(pc[:, None], dR)
    bj, sej, _, _ = fit(np.column_stack([pc, pr]), dR)
    print(f'  control, constant d_wP: slope {bc[1]:.2f}, R2 {r2c:.3f} | joint: constant {bj[1]:.2f} '
          f'(t {bj[1] / sej[1]:.1f}), AERONET {bj[2]:.2f} (t {bj[2] / sej[2]:.1f})')


if __name__ == '__main__':
    d, dR = load_s3('VEIT')
    test('OLCI VEIT Oa17', d, dR.sel(band='Oa17').values)
    d, e = load_s2('VEIT')
    test('MSI VEIT B8A', d, e.sel(wl=865).values)
