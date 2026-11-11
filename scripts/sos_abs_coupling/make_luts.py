"""
SOS-ABS simulations of the TOA signal over a rough sea surface (Cox-Munk) + Lambertian water reflectance,
with full coupling of gaseous absorption (CKD 5 cm-1, fine mode) and scattering, to assess and
parameterise the decoupled gaseous transmittance used by radcalnet_oc.

LUTs written to OUTDIR:
  abs.nc     : TOA field with absorption      (wl, sza, aot, rho)
  noabs.nc   : TOA field without absorption   (same grid)
  tgdir.nc   : direct flux with/without absorption vs sza (air mass 1..10) -> Tg_dir(m) from CKD
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

# OUTDIR and the Mie / surface cache must be on a local disk, not in a synchronised folder (Dropbox):
# 'data' is a link to such a directory. The cache is shared by all the workers; the surface matrix files
# are cached per sza (older sosabs versions reused them across sza: LUTs made with them must be rebuilt).
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE = os.path.join(OUTDIR, "cache")
os.makedirs(CACHE, exist_ok=True)

# absorption windows (nm, step nm)
WINDOWS = [(684, 698, 0.5),    # O2-B
           (712, 736, 0.5),    # H2O 720
           (757, 772, 0.5),    # O2-A
           (808, 836, 0.5),    # H2O 820
           (890, 990, 1.0)]    # H2O 940
WL = np.unique(np.concatenate([np.arange(a, b + 1e-6, s) for a, b, s in WINDOWS])) * 1e-3

ABS = {"-AP.AbsProfile.Type": 2, "-AP.SpectralResol": 5.0, "-SOS.AbsModeCKD": 1, "-AP.H2O": 2.0}
NOABS = {"-AP.AbsProfile.Type": 7}
BASE = {"-ANG.Rad.NbGauss": 24, "-ANG.Aer.NbGauss": 40, "-SOS.View.Dphi": 5, "-SOS.OutputAlt": -1.0,
        "-SOS.Ipolar": 0, "-AP.Psurf": 1013., "-AP.HR": 8.0,
        "-AP.AerProfile.Type": 1, "-AP.AerHS.HA": 2.0,
        "-AER.Model": 1, "-AER.WMO.Model": 2, "-AER.Waref": 0.55, "-AER.AOTref": 0.1,
        "-SURF.Type": 1, "-SURF.Ind": 1.33, "-SURF.Glitter.Wind": 5.0, "-SURF.Alb": 0.0}
GRID = {"-SOS_Main.Wa": WL, "-ANG.Thetas": [30., 60.], "-AER.AOTref": [0.05, 0.2],
        "-SURF.Alb": [0.0, 0.01, 0.03, 0.1]}

# direct transmittance vs air mass (cheap: one order, few angles)
SZA_TG = np.degrees(np.arccos(1 / np.array([1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 8.5, 10.])))
TGBASE = {**BASE, "-ANG.Rad.NbGauss": 8, "-SOS.View.Dphi": 30, "-SOS.IGmax": 1, "-SURF.Type": 0}
TGGRID = {"-SOS_Main.Wa": WL, "-ANG.Thetas": SZA_TG}

jobs = [("tgdir_abs.nc", TGGRID, {**TGBASE, **ABS}),
        ("tgdir_noabs.nc", TGGRID, {**TGBASE, **NOABS}),
        ("noabs.nc", GRID, {**BASE, **NOABS}),
        ("abs.nc", GRID, {**BASE, **ABS})]
for name, grid, fixed in jobs:
    out = os.path.join(OUTDIR, name)
    if os.path.exists(out):
        print("exists:", out); continue
    t0 = time.time()
    n = int(np.prod([len(v) for v in grid.values()]))
    print(f"{name}: {n} runs ...", flush=True)
    sosabs.build_lut(grid=grid, fixed=fixed, outfile=out, n_workers=32, cache_dir=CACHE, verbose=False)
    print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)
