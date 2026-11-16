"""
SOS-ABS simulations of the TOA signal over a rough sea surface (Cox-Munk) + Lambertian water reflectance,
with full coupling of gaseous absorption (CKD 5 cm-1, fine mode) and scattering, to assess and
parameterise the decoupled gaseous transmittance used by radcalnet_oc.

Usage: python make_luts.py [v1|v2] [OUTDIR] [N_WORKERS]

Simulation sets (LUTs written to OUTDIR, default data/ for v1 and data/v2/ for v2):

  v1 (research tutorial of the documentation)
    abs.nc, noabs.nc         TOA field with / without absorption, (wl, sza, aot, albedo); H2O 2 g cm-2
    tgdir_abs/noabs.nc       direct flux with / without absorption vs air mass (1..10) -> Tg_dir(wl, m)
  v2 (extended: water vapour column, solar angle, aerosol load)
    abs.nc                   (wl, H2O, sza, aot, albedo)
    noabs.nc                 (wl, sza, aot, albedo)
    tgdir_abs.nc             (wl, H2O, m)
    tgdir_noabs.nc           (wl, m)
  v3 (364-2500 nm outside 684-990 nm, covered by v2; same grid and files as v2)
    wavelengths at the centres of the 5 cm-1 CKD intervals where the gaseous absorption matters
    (0.02 < Tg < 0.99 for an air mass of 2 and 2 g cm-2 of water vapour, REPTRAN table of radcalnet_oc),
    one interval out of STRIDE per spectral region (ckd_wavelengths). The centres avoid the edges of the
    250 cm-1 CKD files, where SOS-ABS stops (ERROR_3500, e.g. 1.6 um = 6250 cm-1). The CKD coefficients
    start at 364 nm.
"""
import os, sys, time
import numpy as np

# SOS-ABS root (compiled code + Python package sosabs in python/)
ROOT = os.environ.setdefault("SOS_ABS_ROOT", "/DATA/git/vrtc/RadiativeTransferCode-SOS")
try:
    import sosabs  # pip install -e $SOS_ABS_ROOT/python
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import sosabs

HERE = os.path.dirname(os.path.abspath(__file__))
SET = sys.argv[1] if len(sys.argv) > 1 else "v1"
if SET not in ("v1", "v2", "v3"):
    sys.exit(__doc__)
# OUTDIR and the Mie / surface cache must be on a local disk, not in a synchronised folder (Dropbox):
# 'data' is a link to such a directory. The cache is shared by all the workers and the sets; the surface
# matrix files are cached per sza (older sosabs versions reused them across sza: LUTs made with them must
# be rebuilt).
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "" if SET == "v1" else SET)

# v3: spectral regions (nm) and stride over the absorbing CKD intervals of each region
V3_REGIONS = [((364, 684), 11),     # O3 Chappuis, NO2, O2 gamma (628 nm), H2O 590 / 650 nm
              ((990, 1300), 4),     # H2O 1130 nm, O2 1270 nm, CO2
              ((1300, 1500), 2),    # H2O 1380 nm
              ((1500, 1800), 3),    # CO2 1575 / 1605 nm, CH4 1665 nm
              ((1800, 2000), 1),    # H2O 1880 nm (mostly saturated)
              ((2000, 2500), 2)]    # CO2 2010 / 2060 nm, CH4 2300 nm, H2O


def ckd_wavelengths(regions=V3_REGIONS, tg_min=0.02, tg_max=0.99, air_mass=2., h2o=20.):
    """Centres (um) of the absorbing 5 cm-1 CKD intervals, one out of `stride` per region."""
    import xarray as xr
    import radcalnet_oc as radoc
    ckd = xr.open_dataset(os.path.join(ROOT, "fic", "COEFF_CKD", "ckd_5cmm1.nc"))
    nu_min, nu_max = ckd.wavenumber_min.values, ckd.wavenumber_max.values
    gt = radoc.GaseousTransmittance(radoc.LUT().gas_lut, o2_model="reptran")
    gt.pressure, gt.air_mass = 1013., air_mass
    gt.gas_tc["h2o"] = h2o                                 # kg m-2
    tg = gt.get_gaseous_transmittance()
    nu = 1e7 / tg.wl.values
    tgi = np.array([tg.values[(nu >= a) & (nu < b)].mean() if ((nu >= a) & (nu < b)).any() else np.nan
                    for a, b in zip(nu_min, nu_max)])
    lam = 1e7 / (0.5 * (nu_min + nu_max))                  # nm
    out = []
    for (lo, hi), stride in regions:
        idx = np.flatnonzero((lam >= lo) & (lam < hi) & (tgi > tg_min) & (tgi < tg_max))
        idx = idx[np.argsort(lam[idx])][::stride]
        out.append(1e4 / (0.5 * (nu_min[idx] + nu_max[idx])))   # um
    return np.sort(np.concatenate(out))
N_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CACHE = os.path.join(HERE, "data", "cache")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# absorption windows (nm, step nm)
if SET == "v1":
    WINDOWS = [(684, 698, 0.5),    # O2-B
               (712, 736, 0.5),    # H2O 720
               (757, 772, 0.5),    # O2-A
               (808, 836, 0.5),    # H2O 820
               (890, 990, 1.0)]    # H2O 940
else:
    WINDOWS = [(684, 698, 0.5), (712, 736, 1.0), (757, 772, 0.5), (808, 836, 1.0), (890, 990, 1.0)]
WL = np.unique(np.concatenate([np.arange(a, b + 1e-6, s) for a, b, s in WINDOWS])) * 1e-3
if SET == "v3":
    WL = ckd_wavelengths()

ABS = {"-AP.AbsProfile.Type": 2, "-AP.SpectralResol": 5.0, "-SOS.AbsModeCKD": 1, "-AP.H2O": 2.0}
NOABS = {"-AP.AbsProfile.Type": 7}
BASE = {"-ANG.Rad.NbGauss": 24, "-ANG.Aer.NbGauss": 40, "-SOS.View.Dphi": 5, "-SOS.OutputAlt": -1.0,
        "-SOS.Ipolar": 0, "-AP.Psurf": 1013., "-AP.HR": 8.0,
        "-AP.AerProfile.Type": 1, "-AP.AerHS.HA": 2.0,
        "-AER.Model": 1, "-AER.WMO.Model": 2, "-AER.Waref": 0.55, "-AER.AOTref": 0.1,
        "-SURF.Type": 1, "-SURF.Ind": 1.33, "-SURF.Glitter.Wind": 5.0, "-SURF.Alb": 0.0}
if SET == "v1":
    H2O = None
    GRID = {"-SOS_Main.Wa": WL, "-ANG.Thetas": [30., 60.], "-AER.AOTref": [0.05, 0.2],
            "-SURF.Alb": [0.0, 0.01, 0.03, 0.1]}
else:
    H2O = [0.5, 2.0, 4.0]
    GRID = {"-SOS_Main.Wa": WL, "-ANG.Thetas": [20., 40., 55., 70.], "-AER.AOTref": [0.05, 0.15, 0.4],
            "-SURF.Alb": [0.0, 0.03, 0.1]}
GRID_ABS = GRID if H2O is None else {"-SOS_Main.Wa": WL, "-AP.H2O": H2O,
                                     **{k: v for k, v in GRID.items() if k != "-SOS_Main.Wa"}}

# direct transmittance vs air mass (cheap: one order, few angles)
SZA_TG = np.degrees(np.arccos(1 / np.array([1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 8.5, 10.])))
TGBASE = {**BASE, "-ANG.Rad.NbGauss": 8, "-SOS.View.Dphi": 30, "-SOS.IGmax": 1, "-SURF.Type": 0}
TGGRID = {"-SOS_Main.Wa": WL, "-ANG.Thetas": SZA_TG}
TGGRID_ABS = TGGRID if H2O is None else {"-SOS_Main.Wa": WL, "-AP.H2O": H2O, "-ANG.Thetas": SZA_TG}

jobs = [("tgdir_abs.nc", TGGRID_ABS, {**TGBASE, **ABS}),
        ("tgdir_noabs.nc", TGGRID, {**TGBASE, **NOABS}),
        ("noabs.nc", GRID, {**BASE, **NOABS}),
        ("abs.nc", GRID_ABS, {**BASE, **ABS})]
for name, grid, fixed in jobs:
    out = os.path.join(OUTDIR, name)
    if os.path.exists(out):
        print("exists:", out); continue
    t0 = time.time()
    n = int(np.prod([len(v) for v in grid.values()]))
    print(f"{SET} {name}: {n} runs ...", flush=True)
    sosabs.build_lut(grid=grid, fixed=fixed, outfile=out, n_workers=N_WORKERS, cache_dir=CACHE, verbose=False)
    print(f"{SET} {name}: done in {time.time() - t0:.0f} s", flush=True)
