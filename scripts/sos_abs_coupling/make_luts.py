"""
SOS-ABS simulations of the TOA signal over a rough sea surface (Cox-Munk) + Lambertian water reflectance,
with full coupling of gaseous absorption (CKD 5 cm-1, fine mode) and scattering, to assess and
parameterise the decoupled gaseous transmittance used by radcalnet_oc.

Usage: python make_luts.py [v1|v2|v3|v4|v5] [OUTDIR] [N_WORKERS] [--run]

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
  v5 (saturated cores of the H2O 1380 and 1880 nm bands, skipped by v3: Tg < 0.02 for an air mass of 2 and
    2 g cm-2): one CKD interval out of two of 1344-1471 and 1801-1964 nm (88 wavelengths, 16 in MSI B10), same grid
    and files as v3. In these cores the decoupled path reflectance is ~0, while the molecules above the water
    vapour scatter light that is not absorbed: the clear-sky L1C B10 has a floor of ~0.5e-3
    (scripts/s2_toa_matchup/residual_budget.py, ../sos_abs_coupling/check_fullrange_s2.py).
  v4 (robustness of the coupling, roadmap tasks 14, 17, 21): the atmosphere of the package and one condition
    changed at a time. Nominal: OPAC COAV_rh70 aerosols of the TOA LUT (mopsmap phase matrices, -AER.Model 4),
    aerosol scale height 2 km, mid-latitude summer gas profile (type 2, as v2 / v3), 1013 hPa, scalar, wind
    5 m s-1, direct sun glint left out (-SOS.DirectSurf 0, as the path reflectance of radcalnet_oc). Variants:
    WMO maritime (bridge with v2 / v3), the 5 other OPAC models, scale height 0.5, 1 and 4 km, gas profiles 1, 3-6,
    1000 and 1030 hPa, polarisation, wind 2 m s-1, direct glint included. Wavelengths of v2 + v3: one out of
    V4_WL_STRIDE for the nominal set, one out of V4_VARIANT_SUBSAMPLE of these for the variants (a subset of the
    nominal wavelengths); reduced grid (V4_GRID). One LUT per variant in OUTDIR/<variant>/ (abs.nc, noabs.nc and,
    for the variants that change the gas profile or the pressure, tgdir_abs/noabs.nc), dimensions
    (wl, h2o, sza, aot550, albedo). The runs without absorption do not depend on the gas profile (the Rayleigh
    profile is set by -AP.HR): for the gas-profile variants, noabs.nc and tgdir_noabs.nc are taken from the
    nominal set at the wavelengths of the variant (attribute shared_from), the nominal set being run first. With -AER.Model 4, SOS-ABS needs -AER.Waref = -SOS_Main.Wa: the runs are
    made wavelength by wavelength, with the AOT at the wavelength from the OPAC extinction (aot550 Cext/Cext550)
    and the phase matrix of the nearest mopsmap file (50 nm grid), converted to the format of SOS-ABS.
    All the runs of a chunk of V4_CHUNK wavelengths go to one pool of N_WORKERS processes (a grid per
    wavelength alone has only 6-18 runs), and each chunk is written in OUTDIR/<variant>/parts/ as soon as it is
    done: an interrupted set restarts from the last chunk.
    Dry run by default (prints the plan, the checks and the size); add --run to run it. Options:
    --variants=nominal,ha1 (subset of the variants), --n-wl=N (first N wavelengths, for a test), --chunk=N
    (wavelengths per pool, default 16), --wl-range=LO,HI (nominal wavelengths within LO-HI nm only),
    --variant-subsample=N (one nominal wavelength out of N for the variants, default 2; 1 runs the variants at
    all the nominal wavelengths, e.g. for one absorption band with --wl-range).
"""
import os, sys, time
import numpy as np

DRY_RUN = "--run" not in sys.argv
# v4 options: --variants=nominal,ha1 (subset of V4_VARIANTS), --n-wl=N (first N wavelengths, for a test)
OPTS = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a)
sys.argv = [a for a in sys.argv if not a.startswith("--")]

# SOS-ABS root (compiled code + Python package sosabs in python/)
ROOT = os.environ.setdefault("SOS_ABS_ROOT", "/DATA/git/vrtc/RadiativeTransferCode-SOS")
try:
    import sosabs  # pip install -e $SOS_ABS_ROOT/python
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import sosabs

HERE = os.path.dirname(os.path.abspath(__file__))
SET = sys.argv[1] if len(sys.argv) > 1 else "v1"
if SET not in ("v1", "v2", "v3", "v4", "v5"):
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

# ---------------------------------------------------------------------------------------------------------------
# v4: robustness of the coupling (see the docstring)
# ---------------------------------------------------------------------------------------------------------------
V2_WINDOWS = [(684, 698, 0.5), (712, 736, 1.0), (757, 772, 0.5), (808, 836, 1.0), (890, 990, 1.0)]
V4_WL_STRIDE = 2
V4_VARIANT_SUBSAMPLE = int(OPTS.get("variant-subsample", 2))   # variants: one nominal wavelength out of 2 (one out of 4 of v2 + v3)
# parameters that do not change the runs without absorption (-AP.AbsProfile.Type 7): such variants take noabs.nc
# and tgdir_noabs.nc from the nominal set
NOABS_INDEPENDENT = ("-AP.AbsProfile.Type",)
V4_CHUNK = int(OPTS.get("chunk", 16))  # wavelengths per pool (16 x 2 AOT x 18 = 576 runs for abs.nc, a few minutes)
V4_GRID = {"-AP.H2O": [0.5, 2.0, 4.0], "-ANG.Thetas": [20., 45., 70.], "aot550": [0.05, 0.2],
           "-SURF.Alb": [0.0, 0.03]}
MOPSMAP = "/DATA/git/vrtc/mopsmap_tbx/scatmat/ascii/scat_mat_opac_{model}_wl{wl:.1f}.csv"
OPAC_MODELS = ["ARCT_rh70", "COAV_rh70", "DESE_rh70", "MACL_rh70", "URBA_rh70", "WASO_rh0"]
V4_NOMINAL = {"aerosol": "COAV_rh70", "-AP.AerHS.HA": 2.0, "-AP.AbsProfile.Type": 2, "-AP.Psurf": 1013.,
              "-SOS.Ipolar": 0, "-SURF.Glitter.Wind": 5.0, "-SOS.DirectSurf": 0}
V4_VARIANTS = {"nominal": {},
               "wmo_maritime": {"aerosol": "WMO_maritime"},
               **{f"aer_{m}": {"aerosol": m} for m in OPAC_MODELS if m != "COAV_rh70"},
               "ha05": {"-AP.AerHS.HA": 0.5}, "ha1": {"-AP.AerHS.HA": 1.0}, "ha4": {"-AP.AerHS.HA": 4.0},
               **{f"profile{t}": {"-AP.AbsProfile.Type": t} for t in (1, 3, 4, 5, 6)},
               "p1000": {"-AP.Psurf": 1000.}, "p1030": {"-AP.Psurf": 1030.},
               "polar": {"-SOS.Ipolar": 1},
               "wind2": {"-SURF.Glitter.Wind": 2.0},
               "direct_glint": {"-SOS.DirectSurf": 1}}
# throughput of the v3 abs.nc set (4272 runs in 1094 s with 32 workers), worker-seconds per run
V3_WORKER_S_PER_RUN = 1094 * 32 / 4272


def v4_wavelengths():
    """Wavelengths (um) of v2 + v3, one out of V4_WL_STRIDE."""
    v2 = np.unique(np.concatenate([np.arange(a, b + 1e-6, s) for a, b, s in V2_WINDOWS])) * 1e-3
    return np.sort(np.concatenate([v2, ckd_wavelengths()]))[::V4_WL_STRIDE]


def mopsmap_read(model, wl_nm):
    """Cext, Csca and the scattering matrix (angle, F11, -F12/F11, F22/F11, F33/F11) of a mopsmap file."""
    with open(MOPSMAP.format(model=model, wl=wl_nm)) as f:
        head = [next(f) for _ in range(4)]
    cext, csca = (float(h.split(":")[1]) for h in head[:2])
    return cext, csca, np.loadtxt(MOPSMAP.format(model=model, wl=wl_nm), skiprows=4)


def opac_optics(model, wl_um):
    """AOT ratio Cext(wl)/Cext(550), single scattering albedo and nearest mopsmap matrix at wl (um).

    Cext is interpolated in log-log space and the single scattering albedo linearly between the 50 nm files."""
    wl = wl_um * 1000
    lo, hi = 50 * np.floor(wl / 50), 50 * np.ceil(wl / 50)
    (c_lo, s_lo, m_lo), (c_hi, s_hi, m_hi) = mopsmap_read(model, lo), mopsmap_read(model, hi)
    x = 0. if hi == lo else (np.log(wl) - np.log(lo)) / (np.log(hi) - np.log(lo))
    cext = np.exp((1 - x) * np.log(c_lo) + x * np.log(c_hi))
    ssa = (1 - x) * s_lo / c_lo + x * s_hi / c_hi
    return cext / mopsmap_read(model, 550.)[0], ssa, (m_lo if wl - lo <= hi - wl else m_hi)


def write_sos_extdata(path, ssa, matrix):
    """External aerosol file of SOS-ABS (-AER.Model 4); angles from 180 to 0 deg as in its documentation."""
    m = matrix[np.argsort(-matrix[:, 0])]
    with open(path, "w") as f:
        f.write(f"EXTINCTION_COEF : {1.:.5E}\nSCATTERING_COEF : {ssa:.5E}\nNB_LINES : {len(m)}\n")
        f.write("Angle    F11        -F12/F11    F22/F11    F33/F11\n")
        np.savetxt(f, m, fmt=["%8.4f", "%.6E", "%.6E", "%.6E", "%.6E"])


def v4_jobs(wls):
    """(variant, aerosol, file, fixed parameters, grid, aot550 axis, wavelengths, shared_from) of the v4 set.

    `wls`: nominal wavelengths; the variants use one out of V4_VARIANT_SUBSAMPLE. `shared_from` is "nominal" for the
    files of the runs without absorption of the variants that only change NOABS_INDEPENDENT parameters, else None.
    """
    base = {**BASE_V4, "-AP.SpectralResol": 5.0, "-SOS.AbsModeCKD": 1}
    tg_keys = ("-AP.AbsProfile.Type", "-AP.Psurf")
    jobs = []
    for name, change in V4_VARIANTS.items():
        conf = {**V4_NOMINAL, **change}
        var_wls = wls if name == "nominal" else wls[::V4_VARIANT_SUBSAMPLE]
        noabs_shared = name != "nominal" and set(change) <= set(NOABS_INDEPENDENT)
        aerosol = conf.pop("aerosol")
        fixed = {**base, **conf}
        grid = {k: v for k, v in V4_GRID.items() if k != "aot550"}
        files = [("abs.nc", fixed, grid), ("noabs.nc", {**fixed, "-AP.AbsProfile.Type": 7},
                                          {k: v for k, v in grid.items() if k != "-AP.H2O"})]
        if name == "nominal" or any(k in change for k in tg_keys):
            # -SURF.Alb is required by SOS-ABS even without surface (it is a grid dimension of abs/noabs only)
            tgfixed = {**fixed, "-ANG.Rad.NbGauss": 8, "-SOS.View.Dphi": 30, "-SOS.IGmax": 1, "-SURF.Type": 0,
                       "-SURF.Alb": 0.0}
            files += [("tgdir_abs.nc", tgfixed, {"-AP.H2O": V4_GRID["-AP.H2O"], "-ANG.Thetas": list(SZA_TG_V4)}),
                      ("tgdir_noabs.nc", {**tgfixed, "-AP.AbsProfile.Type": 7}, {"-ANG.Thetas": list(SZA_TG_V4)})]
        for fname, fx, gr in files:
            aot_axis = None if fname.startswith("tgdir") else V4_GRID["aot550"]
            shared = "nominal" if noabs_shared and fname.endswith("noabs.nc") else None
            jobs.append((name, aerosol, fname, fx, gr, aot_axis, var_wls, shared))
    return jobs


def aerosol_params(aerosol, wl, aot550, extdir):
    """Aerosol keywords of a run at wl (um): WMO maritime, or an OPAC model through an external file."""
    if aerosol == "WMO_maritime":
        return {"-AER.Model": 1, "-AER.WMO.Model": 2, "-AER.Waref": 0.55, "-AER.AOTref": aot550}
    ratio, ssa, matrix = opac_optics(aerosol, wl)
    path = os.path.join(extdir, f"{aerosol}_wl{wl * 1000:.3f}.txt")
    if not DRY_RUN and not os.path.exists(path):
        write_sos_extdata(path, ssa, matrix)
    return {"-AER.Model": 4, "-AER.ExtData": path, "-AER.Waref": float(wl), "-AER.AOTref": float(aot550 * ratio)}


def v4_checks(wls, jobs):
    """Checks before the runs; returns the list of problems."""
    import subprocess
    problems = []
    status = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True).stdout.strip()
    if status:
        problems.append(f"SOS-ABS has {len(status.splitlines())} uncommitted file(s): commit them so that the "
                        "runs can be traced to a code version (lut attributes)")
    from sosabs import params
    for kw in ("-SOS.DirectSurf", "-AER.ExtData", "-AP.AerHS.HA"):
        if kw not in params.NAME:
            problems.append(f"{kw} unknown to this sosabs version")
    header = open(os.path.join(ROOT, "inc", "SOS.h")).read()
    nmax = int(header.split("#define CTE_MAXNB_ANG_EXT")[1].split()[0])
    nang = max(len(mopsmap_read(m, 550.)[2]) for m in OPAC_MODELS)
    if nang > nmax:
        problems.append(f"mopsmap matrices have {nang} angles > CTE_MAXNB_ANG_EXT = {nmax} (inc/SOS.h)")
    missing = sorted({MOPSMAP.format(model=m, wl=w) for m in OPAC_MODELS
                      for w in np.unique(np.r_[50 * np.floor(wls * 20), 50 * np.ceil(wls * 20), 550.])
                      if not os.path.exists(MOPSMAP.format(model=m, wl=w))})
    if missing:
        problems.append(f"{len(missing)} mopsmap file(s) missing, e.g. {missing[0]}")
    # reading of the converted mopsmap files checked with test_extdata.py (COAV_rh70, 0.8 um, against OSOAA)
    return problems


def v4_main():
    wls = v4_wavelengths()
    if "wl-range" in OPTS:
        lo, hi = (float(x) * 1e-3 for x in OPTS["wl-range"].split(","))
        wls = wls[(wls >= lo) & (wls <= hi)]
    if "n-wl" in OPTS:
        wls = wls[:int(OPTS["n-wl"])]
    if "variants" in OPTS:
        unknown = set(OPTS["variants"].split(",")) - set(V4_VARIANTS)
        if unknown:
            sys.exit(f"unknown variant(s) {sorted(unknown)}; variants: {', '.join(V4_VARIANTS)}")
    jobs = [j for j in v4_jobs(wls) if "variants" not in OPTS or j[0] in OPTS["variants"].split(",")]
    print(f"v4: {len(wls)} wavelengths for the nominal set ({wls.min() * 1000:.0f}-{wls.max() * 1000:.0f} nm, "
          f"one out of {V4_WL_STRIDE} of v2 + v3), {len(wls[::V4_VARIANT_SUBSAMPLE])} for the variants; "
          f"{len({j[0] for j in jobs})} variants, chunks of {V4_CHUNK} wavelengths in one pool of {N_WORKERS} "
          f"workers, output {OUTDIR}")
    total, wsec = 0, 0.
    rows = {}
    for name, aerosol, fname, fixed, grid, aot_axis, var_wls, shared in jobs:
        if shared:
            rows.setdefault(name, []).append(f"{fname} from {shared}")
            continue
        n = len(var_wls) * int(np.prod([len(v) for v in grid.values()])) * (len(aot_axis) if aot_axis else 1)
        cost = (3. if fixed.get("-SOS.Ipolar") else 1.) * (0.2 if fname.startswith("tgdir") else 1.)
        rows.setdefault(name, []).append(f"{fname} {n}")
        total += n
        wsec += n * V3_WORKER_S_PER_RUN * cost
    for name, files in rows.items():
        print(f"  {name:16s} " + ", ".join(files))
    print(f"total {total} runs; about {wsec / N_WORKERS / 3600:.1f} h with {N_WORKERS} workers (v3 throughput; "
          "polarised runs x3 and direct-transmittance runs x0.2 assumed)")
    print("checks:")
    for p in v4_checks(wls, jobs):
        print("  -", p)
    if DRY_RUN:
        print("dry run: nothing written (add --run to run, with a memory cap, e.g. "
              "systemd-run --user --scope -p MemoryMax=...)")
        return
    import xarray as xr
    from sosabs.lut import write_netcdf
    extdir = os.path.join(CACHE, "extdata_v4")
    os.makedirs(extdir, exist_ok=True)
    for name, aerosol, fname, fixed, grid, aot_axis, wls, shared in jobs:
        out = os.path.join(OUTDIR, name, fname)
        if os.path.exists(out):
            print("exists:", out)
            continue
        if shared:
            src = os.path.join(OUTDIR, shared, fname)
            if not os.path.exists(src):
                print(f"v4 {name} {fname}: needs {src} (run the {shared} set first), skipped", flush=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with xr.open_dataset(src) as ds:
                sub = ds.sel(wl=wls).load()
            sub.attrs.update({"v4_variant": name, "shared_from": shared})
            write_netcdf(sub, out)
            print(f"v4 {name} {fname}: taken from {shared} ({len(wls)} wavelengths)", flush=True)
            continue
        parts_dir = os.path.join(OUTDIR, name, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        t0, chunk_files = time.time(), []
        for c in range(0, len(wls), V4_CHUNK):
            chunk = wls[c:c + V4_CHUNK]
            cfile = os.path.join(parts_dir, f"{fname[:-3]}_chunk{c // V4_CHUNK:03d}.nc")
            chunk_files.append(cfile)
            if os.path.exists(cfile):
                continue
            parts = []
            for wl in chunk:
                for aot in (aot_axis or [V4_GRID["aot550"][0]]):
                    wl, aot = float(wl), float(aot)       # plain floats for the command line (not np.float64)
                    fx = {**fixed, "-SOS_Main.Wa": wl, **aerosol_params(aerosol, wl, aot, extdir)}
                    coords = {"wl": [wl], **({"aot550": [aot]} if aot_axis else {})}
                    parts.append((coords, grid, fx))
            ds = run_parts(parts, attrs={"v4_variant": name, "aerosol": aerosol})
            write_netcdf(ds, cfile)
            print(f"v4 {name} {fname}: chunk {c // V4_CHUNK + 1}/{-(-len(wls) // V4_CHUNK)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        lut = xr.combine_by_coords([xr.open_dataset(f).load() for f in chunk_files],
                                   combine_attrs="drop_conflicts")
        lut.attrs.update({"v4_variant": name, "aerosol": aerosol})
        write_netcdf(lut, out)
        print(f"v4 {name} {fname}: done in {time.time() - t0:.0f} s", flush=True)


def run_parts(parts, attrs=None):
    """Run several grids of SOS-ABS in one pool and gather them in one Dataset.

    `parts`: list of (coords, grid, fixed); each grid is a LUT of :func:`sosabs.build_lut` with its own fixed
    parameters (here the wavelength, its aerosol file and AOT), and `coords` the dimensions added to it (wl,
    aot550). The runs of all the parts share one pool of N_WORKERS processes and the cache of Mie and surface
    files (safe: the code moves each file into the cache by an atomic rename). Uses the internal functions of
    sosabs.lut (_run_job, _collect, _assemble), as build_lut does for one grid.
    """
    import itertools
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor
    import xarray as xr
    from sosabs import params as P, runner
    from sosabs.lut import _run_job, _collect, _assemble
    t0 = time.time()
    root = runner.sos_root(None)
    os.environ["SOS_ABS_ROOT"] = root
    backend = runner.available_backend(root)
    os.makedirs(CACHE, exist_ok=True)
    jobs, meta = [], []
    for k, (coords, grid, fixed) in enumerate(parts):
        grid = P.normalize(grid)
        fixed = {n: v for n, v in P.normalize(fixed).items() if n not in grid}
        names = list(grid)
        values = [np.atleast_1d(np.asarray(grid[n])) for n in names]
        shape = tuple(len(v) for v in values)
        meta.append((coords, names, values, shape, fixed))
        for idx in itertools.product(*[range(n) for n in shape]):
            par = dict(fixed)
            for n, v, i in zip(names, values, idx):
                par[n] = v[i].item()
            jobs.append(((k, idx), par, backend, root, CACHE))
    results = {}
    n_workers = max(1, min(N_WORKERS, len(jobs)))
    if backend == "f2py":   # one process per run (fresh Fortran state), as build_lut
        with mp.get_context("fork").Pool(n_workers, maxtasksperchild=1) as pool:
            _collect(pool.imap_unordered(_run_job, jobs), results, len(jobs), False, t0)
    else:
        with ThreadPoolExecutor(n_workers) as ex:
            _collect(ex.map(_run_job, jobs), results, len(jobs), False, t0)
    out = []
    for k, (coords, names, values, shape, fixed) in enumerate(meta):
        res = {idx: r for (kk, idx), r in results.items() if kk == k}
        ds = _assemble(names, values, shape, res, True, fixed, backend, root, time.time() - t0, attrs)
        if "aot_ref" in fixed:      # AOT at the wavelength of the run (-AER.AOTref, -AER.Waref = wl)
            ds["aot_wl"] = float(fixed["aot_ref"])
            ds["aot_wl"].attrs = {"long_name": "aerosol optical thickness at the wavelength of the run"}
        for dim, val in coords.items():
            ds = ds.drop_vars(dim, errors="ignore").expand_dims({dim: val})
        out.append(ds)
    return xr.combine_by_coords(out, combine_attrs="drop_conflicts")

BASE_V4 = {"-ANG.Rad.NbGauss": 24, "-ANG.Aer.NbGauss": 40, "-SOS.View.Dphi": 5, "-SOS.OutputAlt": -1.0,
           "-AP.HR": 8.0, "-AP.AerProfile.Type": 1, "-SURF.Type": 1, "-SURF.Ind": 1.33}
SZA_TG_V4 = np.degrees(np.arccos(1 / np.array([1., 1.5, 2., 3., 4., 6., 8.5])))
if SET == "v4":
    OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "v4")
    v4_main()
    sys.exit(0)

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
#: v5: saturated cores of the H2O 1380 and 1880 nm bands (one CKD interval out of two)
V5_REGIONS = [((1300, 1500), 2), ((1800, 2000), 2)]
if SET == "v5":
    WL = ckd_wavelengths(regions=V5_REGIONS, tg_min=0., tg_max=0.02)

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
