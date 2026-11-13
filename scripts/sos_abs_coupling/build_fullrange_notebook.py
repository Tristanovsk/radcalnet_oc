"""Build notebook/research/radcalnet_oc_gaseous_coupling_350_2500nm.ipynb (study of the gaseous
absorption-scattering coupling over 350-2500 nm, simulations v2 + v3 of make_luts.py).

Usage: python build_fullrange_notebook.py [OUTPUT.ipynb] [CONCLUSIONS.md]
"""
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, '..', '..', 'notebook', 'research', 'radcalnet_oc_gaseous_coupling_350_2500nm.ipynb')
CONCLUSIONS = open(sys.argv[2]).read() if len(sys.argv) > 2 else '*To be completed after the execution.*'

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r"""
# Gaseous absorption and scattering over 350-2500 nm: coupling and its parameterization in RadCalNet-OC

RadCalNet-OC computes the TOA signal as the product of the scattering terms (path reflectance $R_{atm}$,
water-leaving term, downwelling irradiance $E_d$) and of gaseous transmittances. In the decoupled model each
term is attenuated by the **direct-beam** transmittance $T_g(m)$, with $m = 1/\mu_0 + 1/\mu_v$ (or $1/\mu_0$ for
$E_d$), which ignores that scattered photons cross the absorbers along other paths. The study
`radcalnet_oc_gaseous_coupling_analysis.ipynb` quantified this for the O$_2$ and H$_2$O bands of 684-990 nm and led
to the photon path-length model (`gas_coupling='path_length'`, default) in seven windows of that range;
**outside these windows the package still uses the decoupled model**, and CH$_4$, O$_3$ and NO$_2$ are never coupled.

This notebook extends the study to the whole spectral range of the package, **350-2500 nm**:

1. where does the coupling matter, for which term and which absorber?
2. how do the photon paths through the absorbers depend on the vertical distribution of the absorber
   (stratospheric O$_3$, well-mixed O$_2$ / CO$_2$ / CH$_4$, H$_2$O in the boundary layer) and on the scattering
   regime (Rayleigh in the visible, aerosols in the shortwave infrared)?
3. does the photon path-length model hold over the whole range, and how compact can its parameterization be?
4. what should RadCalNet-OC implement, at which cost?

**Reference.** SOS-ABS (successive orders of scattering with correlated-k absorption, `-SOS.AbsModeCKD 1`: the
radiative transfer is solved for every combination of the k-terms of the eight gases, i.e. exact coupling at the
5 cm$^{-1}$ resolution of the CKD), ocean surface (Cox-Munk, 5 m s$^{-1}$) with a Lambertian water reflectance,
WMO maritime aerosols (scale height 2 km), mid-latitude summer gas profiles (O$_3$ 333 DU, CO$_2$ 330 ppmv,
CH$_4$ 1.7 ppmv, H$_2$O rescaled to the column of the grid).

* **v2** (`make_luts.py v2`): 684-990 nm, every 0.5-1 nm (the set of the 684-990 nm study);
* **v3** (`make_luts.py v3`): 364-2500 nm outside 684-990 nm, at the centres of the 5 cm$^{-1}$ CKD intervals
  where the absorption matters ($0.02 < T_g < 0.99$ for $m = 2$ and 2 g cm$^{-2}$ of water vapour), one interval out
  of 1-11 depending on the region (463 wavelengths).

Same grid for both: water vapour 0.5, 2, 4 g cm$^{-2}$; solar zenith angle 20, 40, 55, 70°; AOT(550) 0.05, 0.15,
0.4; water albedo 0, 0.03, 0.1 (to separate the terms); viewing zenith angles up to 60° and all azimuths
(geometries within 30° of the specular direction left out).

**Coverage.** The CKD coefficients of SOS-ABS start at 364 nm (27500 cm$^{-1}$): between 350 and 364 nm the only
absorber is the O$_3$ Huggins band (transmittance > 0.97). The CKD has no O$_2$-O$_2$ collision-induced absorption
(O$_4$ bands at 477, 577, 630 and 1060 nm, < 2 % in the direct beam), so these bands are not assessed.

The computations use `scripts/sos_abs_coupling/fullrange.py` (with `coupling.py` and `ppld.py` of the 684-990 nm
study).
""")

code(r"""
import os
import sys
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '../../scripts/sos_abs_coupling')
import coupling as cp
import fullrange as fr
import ppld
import radcalnet_oc as radoc

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'figure.dpi': 90})
pd.set_option('display.width', 200)
TERMS = {'atm': 'path reflectance', 'w': 'water-leaving term', 'Ed': 'irradiance'}
CLASS_COLORS = {'O3 (stratospheric)': 'tab:purple', 'well mixed (O2, CO2, CH4)': 'tab:green',
                'H2O (boundary layer)': 'tab:blue'}
print('radcalnet_oc', radoc.__version__)
""")

md(r"""
## 1. Gaseous absorption over 350-2500 nm

Direct-beam transmittance of each gas for an air mass of 2 (gaseous table of RadCalNet-OC, default columns,
2 g cm$^{-2}$ of water vapour), with the windows of the study (grey: the 684-990 nm windows of the package) and the
simulated wavelengths.
""")

code(r"""
gt = radoc.GaseousTransmittance(radoc.LUT().gas_lut, o2_model='reptran')
gt.pressure, gt.air_mass = 1013., 2.
gt.gas_tc['h2o'] = 20.
tg_gas = {g: gt.Tgas(g) for g in ('h2o', 'o3', 'no2', 'ch4')}
tg_gas['O2, CO2, O4, CO'] = gt.Tgas_background()
wl_sim = {'v2': np.round(np.unique(np.concatenate([np.arange(a, b + 1e-6, s) for (a, b), s in fr.V2_STEPS])), 2),
          'v3': np.load(os.path.join(fr.DATA_V3, 'wavelengths.npy')) if os.path.exists(os.path.join(fr.DATA_V3, 'wavelengths.npy'))
          else np.round(xr.open_dataset(os.path.join(fr.DATA_V3, 'noabs.nc')).wl.values * 1e3, 2)}

fig, ax = plt.subplots(figsize=(17, 6))
cols = {'h2o': 'tab:blue', 'o3': 'tab:purple', 'no2': 'tab:brown', 'ch4': 'tab:red', 'O2, CO2, O4, CO': 'tab:green'}
for g, t in tg_gas.items():
    ax.plot(t.wl, t.rolling(wl=10, center=True).mean(), lw=0.6, color=cols[g], label=g.upper() if len(g) < 5 else g)
for w, (a, b) in fr.WINDOWS_FULL.items():
    ax.axvspan(a, b, color='0.85' if w in fr.WINDOWS_684_990 else '0.94', zorder=-2)
    if w in fr.WINDOWS_NEW:
        ax.text((a + b) / 2, 0.02, w, rotation=90, ha='center', va='bottom', fontsize=9)
ax.plot(wl_sim['v2'], np.full(len(wl_sim['v2']), 1.03), '|', color='k', ms=6, label='simulations v2')
ax.plot(wl_sim['v3'], np.full(len(wl_sim['v3']), 1.06), '|', color='tab:orange', ms=6, label='simulations v3')
ax.set(xlim=(350, 2500), ylim=(0, 1.09), xlabel='wavelength (nm)', ylabel=r'$T_g(m=2)$ (0.2 nm, smoothed to 2 nm)')
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=11)
plt.show()
""")

md(r"""
## 2. Simulations and effective transmittances

The TOA reflectance of each simulation is decomposed as $R(\rho) = R_0 + \rho A / (1 - S\rho)$ (three albedos), the
direct sun glint being removed from $R_0$ as in the LUT of RadCalNet-OC. The effective transmittances of the three
terms are the ratios with / without absorption:

$$T_{atm} = \frac{R_0^{abs}}{R_0^{noabs}}, \qquad T_w = \frac{A^{abs}}{A^{noabs}}, \qquad
T_{E_d} = \frac{E_d^{abs}}{E_d^{noabs}},$$

to be compared with the direct transmittance $T_g(m)$ of the decoupled model, from the direct fluxes of SOS-ABS for
air masses 1-10 (same CKD). The channels kept are those with $T_g(m=2) < 0.999$ (the O$_4$ intervals of the selection
are transparent in SOS-ABS).
""")

code(r"""
t0 = time.time()
eff = fr.load_full()
a, rms_es = ppld.exponential_sum(eff['tg'])
fr.use_full_windows()
wl = eff['T'].wl.values
dom, tau_gas = fr.dominant_gas(wl)
print(f'loaded in {time.time() - t0:.0f} s: {len(wl)} absorbing channels '
      f'({int((eff["set"] == "v2").sum())} v2, {int((eff["set"] == "v3").sum())} v3); '
      f'exponential sums of Tg(m): max rms {float(rms_es.max()):.1e}')
summary = pd.DataFrame({'window': eff['window'].values, 'class': [fr.CLASS.get(w, '') for w in eff['window'].values],
                        'dominant gas': dom.values, 'Tg(m=2)': cp.tg_of_m(eff['tg'], 2.).sel(h2o=2.).values}, index=wl)
summary.groupby('window', sort=False).agg(channels=('class', 'size'), wl_min=('Tg(m=2)', lambda x: x.index.min()),
                                          wl_max=('Tg(m=2)', lambda x: x.index.max()),
                                          gases=('dominant gas', lambda x: ', '.join(x.value_counts().index[:3])),
                                          Tg_min=('Tg(m=2)', 'min'), Tg_median=('Tg(m=2)', 'median')).round(3)
""")

md(r"""
### Absorption tables: RadCalNet-OC against the CKD of SOS-ABS

The coupling corrections are relative ($T/T_g$); their transfer to the package also depends on the agreement of
the absorption tables themselves. Absorption depth $1 - T_g$ for $m = 2$ and 2 g cm$^{-2}$ of water vapour on the
sampled 5 cm$^{-1}$ intervals (RadCalNet-OC: REPTRAN at 0.2 nm averaged on the interval, default columns: 1 for
O$_2$ / CO$_2$ scaling, CH$_4$ 1e-2 kg m$^{-2}$, O$_3$ 6.5e-3 kg m$^{-2}$ = 303 DU; SOS-ABS: CO$_2$ 330 ppmv,
O$_3$ 333 DU).
""")

code(r"""
nu = 1e7 / gt.gas_lut.wl.values
tg_pkg = gt.get_gaseous_transmittance()
pkg = np.array([float(tg_pkg.values[(nu >= 1e7 / w - 2.5) & (nu < 1e7 / w + 2.5)].mean()) for w in wl])
ckd = cp.tg_of_m(eff['tg'], 2.).sel(h2o=2.).values
ratio = (1 - pkg) / (1 - ckd)
fig, ax = plt.subplots(figsize=(17, 4.5))
for c, col in CLASS_COLORS.items():
    s = summary['class'].values == c
    ax.semilogy(wl[s], ratio[s], '.', color=col, ms=4, label=c)
ax.axhline(1, color='k', lw=0.8)
ax.set(xlim=(350, 2500), ylim=(0.1, 10), xlabel='wavelength (nm)', ylabel='absorption depth RadCalNet-OC / CKD')
ax.legend()
plt.show()
tab_ratio = pd.DataFrame({'window': summary.window, 'ratio': ratio, 'w': 1 - ckd})
tab_ratio.groupby('window', sort=False).apply(lambda d: pd.Series({
    'median ratio': d.ratio.median(), 'absorption-weighted ratio': (d.ratio * d.w).sum() / d.w.sum()})).round(2).T
""")

md(r"""
## 3. Effective path lengths through the absorbers

The effective air-mass ratio $\eta = m^*/m$, with $m^*$ such that $T_g(m^*) = T_X$, is the mean length of the
photon paths through the absorber relative to the geometric path of the decoupled model ($\eta = 1$: decoupled
model exact). Median over the conditions (and IQR) per channel, for the channels with $T_g(m) < 0.95$:
""")

code(r"""
fig, axs = plt.subplots(3, 1, figsize=(17, 12), sharex=True)
cls = np.array([fr.CLASS.get(w, '') for w in eff['window'].values])
for ax, (term, lab) in zip(axs, TERMS.items()):
    e = eff['eta'][term]
    dims = [d for d in e.dims if d != 'wl']
    med, q1, q3 = e.median(dims), e.quantile(0.25, dims), e.quantile(0.75, dims)
    for c, col in CLASS_COLORS.items():
        s = cls == c
        ax.errorbar(wl[s], med.values[s], yerr=[med.values[s] - q1.values[s], q3.values[s] - med.values[s]],
                    fmt='.', ms=3, lw=0.4, color=col, label=c)
    ax.axhline(1, color='k', lw=0.8)
    ax.set(ylabel=r'$\eta$', title=lab, ylim=(0, 1.6))
axs[0].legend(ncol=3)
axs[-1].set(xlim=(350, 2500), xlabel='wavelength (nm)')
plt.show()
""")

code(r"""
rows = []
for w in fr.WINDOWS_FULL:
    s = eff['window'] == w
    if not s.any():
        continue
    r = {'window': w, 'class': fr.CLASS[w]}
    for term in TERMS:
        x = eff['eta'][term].isel(wl=np.flatnonzero(s.values))
        r[(term, 'median')] = float(x.median())
        r[(term, 'IQR')] = float(x.quantile(0.75) - x.quantile(0.25))
        r[(term, 'n')] = int(x.count())
    rows.append(r)
eta_tab = pd.DataFrame(rows).set_index(['window', 'class'])
eta_tab.columns = pd.MultiIndex.from_tuples(eta_tab.columns)
eta_tab.round(3)
""")

md(r"""
## 4. Error of the decoupled model

Two metrics:

* **channel** (5 cm$^{-1}$ interval, i.e. hyperspectral resolution): absolute transmittance error
  $\Delta T = T_{model} - T_X$, in % of the signal without absorption;
* **band**: error of the absorption depth of a band, in % of the absorption,
  $\delta_b = \sum_i w_i S_i (T_{model,i} - T_{X,i}) / \sum_i w_i S_i (1 - T_{X,i})$, with $w_i$ the band response
  times the solar irradiance times the spectral interval represented by the sample, and $S_i$ the signal of the term
  without absorption. The transparent intervals, not simulated, add neither absorption nor error. The band TOA
  error of the term is about $\delta_b$ times the band absorption depth $D_b$.
""")

code(r"""
def channel_stats(models, windows=None):
    rows = []
    for w in windows or fr.WINDOWS_FULL:
        s = np.flatnonzero((eff['window'] == w).values)
        if not len(s):
            continue
        r = {'window': w}
        for term in TERMS:
            for name, T in models.items():
                if T[term] is None:
                    continue
                x = abs(fr.channel_error(eff, T[term], term).isel(wl=s)) * 100
                r[(term, name, 'median')] = float(x.median())
                r[(term, name, 'p95')] = float(x.quantile(0.95))
        rows.append(r)
    t = pd.DataFrame(rows).set_index('window')
    t.columns = pd.MultiIndex.from_tuples(t.columns)
    return t


T_dec = {term: fr.decoupled(eff, term) for term in TERMS}
channel_stats({'decoupled': T_dec}).round(2)
""")

md(r"""
## 5. Photon path-length model over the full range

The model of the package (`radcalnet_oc.kernel.path_length_transmittance`) describes the distribution of the absorber
paths of the photons of a term by a Gamma distribution of mean $\eta m$ and relative variance $s^2$; by the
equivalence theorem, for a monochromatic absorber of optical path $\tau$ along the geometric air mass,

$$T_X = (1 + s^2 \eta \tau)^{-1/s^2},$$

applied to each exponential of the CKD intervals. $\eta$ and $s^2$ are linear in $(m - 2,\ \tau_a(550) - 0.1,\ \ln U/2)$
per window and term (8 coefficients, 6 without the water vapour term in the O$_3$ window), fitted on the absolute
transmittance error. The coefficients of the seven 684-990 nm windows are refitted with the others (same data as the
table of the package).
""")

code(r"""
t0 = time.time()
model = fr.fit_full(eff, a)
print(f'fit of {len(model["coefs"])} (window, term) models in {time.time() - t0:.0f} s')
T_ppld = {term: fr.apply_full(eff, model, a, term) for term in TERMS}
rows = []
for (w, term), (ce, cs) in model['coefs'].items():
    rows.append({'window': w, 'class': fr.CLASS[w], 'term': term, 'eta_ref': ce[0], 's2_ref': max(cs[0], ppld.S2_MIN),
                 'd eta / d m': ce[1], 'd eta / d aot': ce[2], 'd eta / d lnU': ce[3], 'rms (%)': model['rms'][(w, term)] * 100})
coef_tab = pd.DataFrame(rows)
coef_tab.pivot_table(index=['window', 'class'], columns='term', values=['eta_ref', 's2_ref', 'rms (%)'], sort=False).round(3)
""")

code(r"""
fig, axs = plt.subplots(1, 2, figsize=(17, 5))
centre = {w: np.mean(fr.WINDOWS_FULL[w]) for w in fr.WINDOWS_FULL}
mk = {'atm': 'o', 'w': 's', 'Ed': '^'}
for term in TERMS:
    d = coef_tab[coef_tab.term == term]
    for c, col in CLASS_COLORS.items():
        dd = d[d['class'] == c]
        x = [centre[w] for w in dd.window]
        axs[0].plot(x, dd.eta_ref, mk[term], color=col, ms=7, mfc='none' if term != 'atm' else col)
        axs[1].plot(x, dd.s2_ref, mk[term], color=col, ms=7, mfc='none' if term != 'atm' else col)
for ax, lab in zip(axs, (r'$\eta$ (m = 2, AOT 0.1, U = 2 g cm$^{-2}$)', r'$s^2$')):
    ax.axhline(1 if lab.startswith(r'$\eta') else 0, color='k', lw=0.8)
    ax.set(xlabel='window centre (nm)', ylabel=lab, xlim=(350, 2500))
handles = [mpl.lines.Line2D([], [], color=col, marker='o', ls='', label=c) for c, col in CLASS_COLORS.items()]
handles += [mpl.lines.Line2D([], [], color='k', marker=mk[t], ls='', mfc='none' if t != 'atm' else 'k', label=TERMS[t]) for t in TERMS]
axs[0].legend(handles=handles, fontsize=10)
axs[1].semilogy()
plt.show()
""")

code(r"""
channel_stats({'decoupled': T_dec, 'ppld': T_ppld}).round(2)
""")

md(r"""
### Band level: 10 nm bands across the range

Absorption-depth error $\delta_b$ (median and 95th percentile over the conditions) of Gaussian bands of 10 nm FWHM
centred on the simulated windows, for the decoupled and the photon path-length models.
""")

code(r"""
centres = [c for c in np.arange(380, 2500, 10) if (abs(wl - c) < 7).sum() >= 2]


def sweep(models, term, fwhm=10.):
    out = {k: [] for k in models}
    for c in centres:
        for k, T in models.items():
            x = abs(fr.depth_error(eff, T[term], term, c, fwhm))
            out[k].append((float(x.median()), float(x.quantile(0.95))))
    return {k: np.array(v) for k, v in out.items()}


fig, axs = plt.subplots(3, 1, figsize=(17, 12), sharex=True)
sweeps = {}
for ax, term in zip(axs, TERMS):
    sweeps[term] = s = sweep({'decoupled': T_dec, 'ppld': T_ppld}, term)
    for k, col in (('decoupled', 'tab:orange'), ('ppld', 'tab:blue')):
        ax.semilogy(centres, s[k][:, 0], 'o-', color=col, ms=3, lw=0.8, label=f'{k}, median')
        ax.semilogy(centres, s[k][:, 1], ':', color=col, lw=0.8, label=f'{k}, 95th percentile')
    ax.set(ylabel=r'$|\delta_b|$ (% of the absorption)', title=TERMS[term], ylim=(0.01, 300))
    ax.grid(which='major', alpha=0.3)
axs[0].legend(ncol=4)
axs[-1].set(xlabel='band centre (nm)', xlim=(350, 2500))
plt.show()
""")

md(r"""
### Robustness: cross-validation and a compact parameterization

* **leave-one-out**: the model is refitted without the water vapour column 4 g cm$^{-2}$ (resp. the solar zenith
  angle 70°) and evaluated on it (extrapolation);
* **per absorber class**: a single model per class of vertical distribution (O$_3$; well-mixed gases; H$_2$O) for
  all its windows, instead of one per window.
""")

code(r"""
t0 = time.time()
cv = {}
for name, excl in (('H2O 4 g cm-2', {'h2o': 4.0}), ('SZA 70', {'sza': 70.0})):
    m_cv = fr.fit_full(eff, a, exclude=excl, n=30000)
    cv[name] = (excl, {term: fr.apply_full(eff, m_cv, a, term) for term in TERMS})
coefs_class = fr.fit_class(eff, a, n=30000)
T_class = {term: fr.apply_full(eff, model, a, term, coefs=coefs_class) for term in TERMS}
print(f'{time.time() - t0:.0f} s')

rows = []
for w in fr.WINDOWS_FULL:
    s = np.flatnonzero((eff['window'] == w).values)
    if not len(s):
        continue
    r = {'window': w}
    for term in ('atm', 'Ed'):
        err = lambda T, only=None: float((abs(cp.subset(fr.channel_error(eff, T[term], term).isel(wl=s), only=only)) * 100).median())
        r[(term, 'ppld')] = err(T_ppld)
        for name, (excl, T) in cv.items():
            r[(term, f'CV {name}')] = err(T, only=excl)
            r[(term, f'ppld on {name}')] = err(T_ppld, only=excl)
        r[(term, 'per class')] = err(T_class)
        r[(term, 'decoupled')] = err(T_dec)
    rows.append(r)
cv_tab = pd.DataFrame(rows).set_index('window')
cv_tab.columns = pd.MultiIndex.from_tuples(cv_tab.columns)
cv_tab.round(2)
""")

md(r"""
## 6. Sensor bands

Absorption-depth error $\delta_b$ (median [95th percentile] over the conditions, % of the absorption) of nominal
Gaussian approximations of sensor bands in absorption features, for the path reflectance and the irradiance; $D_b$ is
the absorption depth of the band for the direct beam ($m = 2$, 2 g cm$^{-2}$), and $\delta_b D_b$ the order of the band
TOA error of the term.
""")

code(r"""
BANDS = {'MSI B3 (560)': (560, 35), 'MSI B4 (665)': (665, 30), 'OLCI Oa13 (761)': (761.25, 2.5),
         'OLCI Oa15 (768)': (767.5, 2.5), 'OLCI Oa19 (900)': (900, 10), 'OLCI Oa20 (940)': (940, 20),
         'OLCI Oa21 (1016)': (1016, 40), 'MSI B9 (945)': (945, 20), 'OCI 1038': (1038, 75), 'VIIRS M8 (1240)': (1240, 20),
         'OCI 1250': (1250, 30), 'MSI B10 (1375)': (1375, 30), 'OLI B9 (1373)': (1373, 21), 'OCI 1378': (1378, 15),
         'MSI B11 (1610)': (1610, 90), 'OLI B6 (1609)': (1609, 85), 'VIIRS M10 (1610)': (1610, 60),
         'OCI 2130': (2130, 50), 'MSI B12 (2190)': (2190, 180), 'OLI B7 (2201)': (2201, 187), 'OCI 2260': (2260, 75)}
rows = []
for b, (c, f) in BANDS.items():
    D = fr.band_depth(c, f)
    r = {'band': b, 'D_b (%)': D * 100}
    for term in ('atm', 'Ed'):
        for k, T in (('decoupled', T_dec), ('ppld', T_ppld), ('per class', T_class)):
            x = abs(fr.depth_error(eff, T[term], term, c, f))
            r[(term, k)] = f'{float(x.median()):.1f} [{float(x.quantile(0.95)):.1f}]'
            if k != 'per class':
                r[(term, f'{k} x D_b')] = float(x.median()) * D
    rows.append(r)
band_tab = pd.DataFrame(rows).set_index(['band', 'D_b (%)'])
band_tab.columns = pd.MultiIndex.from_tuples([c if isinstance(c, tuple) else (c, '') for c in band_tab.columns])
band_tab.round(2)
""")

md(r"""
## 7. Implications for RadCalNet-OC

The package corrects, in each window of the coupling table, the transmittance of the water vapour and of the
background gases (O$_2$, O$_4$, CO$_2$, CO), CH$_4$, O$_3$ and NO$_2$ keeping the geometric air mass
(`GaseousTransmittance.COUPLED_GASES`, `UNCOUPLED_GASES`). The candidate table of the photon path-length model for the
whole range, in the format of `data/auxdata/gas_coupling_ppld.csv`, is written to
`scripts/sos_abs_coupling/gas_coupling_ppld_full.csv` (not used by the package).
""")

code(r"""
table = fr.table_frame(model)
table.to_csv(fr.TABLE_FULL, index=False, float_format='%.6g')
print('written', os.path.relpath(fr.TABLE_FULL))
table[['window', 'wl_min', 'wl_max', 'term', 'c0', 'c1', 'c2', 'c3', 's0', 's1', 's2', 's3', 'rms']].round(4)
""")

md(r"""
Cost: the coupled transmittance is computed per window on the channels of the window; timing of the three terms of
the package on the clear-water template with the present table (7 windows) and with one window per new region
(18 windows, coefficients of the candidate table):
""")

code(r"""
from radcalnet_oc import GasCoupling
from radcalnet_oc.kernel import GAS_COUPLING_PPLD_FILE

geo_m = xr.DataArray(np.linspace(2, 6, 20), dims='geom')
timing = {}
for name, cpl in (('7 windows (package)', GasCoupling(GAS_COUPLING_PPLD_FILE)), ('18 windows (candidate)', GasCoupling(table))):
    g = radoc.GaseousTransmittance(radoc.LUT().gas_lut)
    g.pressure = 1013.
    g.gas_tc['h2o'] = 20.
    t0 = time.time()
    for term in TERMS:
        g.get_coupled_transmittance(geo_m, term, 0.1, cpl)
    timing[name] = time.time() - t0
pd.Series(timing, name='time (s), 3 terms, 20 geometries').round(2).to_frame()
""")

md(r"""
## 8. Conclusions

%%CONCLUSIONS%%
""".replace('%%CONCLUSIONS%%', CONCLUSIONS))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, OUT)
print('written', OUT, len(cells), 'cells')
