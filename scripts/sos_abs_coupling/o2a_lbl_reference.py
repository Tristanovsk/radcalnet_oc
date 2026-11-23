"""
Line-resolved SOS-ABS reference of the O2-A coupling: is the correlated-k (CKD) coupling of SOS-ABS, on which the
ppld table of radcalnet_oc is fitted, right for the scattered light in the O2 lines?

SOS-ABS has no line-by-line mode. Monochromatic runs are made through its CKD files: in shadow roots of SOS-ABS
(symbolic links to the installation, $SOS_ABS_ROOT/fic/COEFF_CKD/1cmm1 replaced), the O2 file of each 1 cm-1
interval of the O2-A band has one term (a = 1) whose k(P, T) is the Voigt cross section of the HITRAN 2012 O2
lines at one wavenumber of the interval (stratified random sampling, one wavenumber per interval and per root);
the other gases are set to no absorption in these files. A run at the wavelength of the sample is then
monochromatic, with the exact correlation of k between the levels. A CKD root (original O2 coefficients at
1 cm-1, other gases off) gives the CKD result on the same intervals.

Settings of the v4 nominal set (make_luts.py): OPAC COAV_rh70, scale height 2 km, mid-latitude summer, 1013 hPa,
scalar, wind 5 m s-1, direct glint off; sza 20/45/70 deg, aot550 0.05/0.2, albedo 0 (path reflectance R0).
Direct transmittance from the direct flux of SOS-ABS with the same files (air masses of SZA_TG_V4).

Usage:
    python o2a_lbl_reference.py files [--samples=4]           # line cross sections, CKD files, shadow roots
    python o2a_lbl_reference.py run [--n-workers=12] [--only=ckd,s0] [--n-wl=N]
    python o2a_lbl_reference.py analyse

Output in /data/sos_abs_coupling/v4_o2a/lbl_ref: roots/<set>/ (shadow roots), <set>/{abs,tgdir_abs}.nc,
noabs.nc, tgdir_noabs.nc, fpath.csv.
"""
import os, sys, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOS = "/DATA/git/vrtc/RadiativeTransferCode-SOS"
OUT = "/data/sos_abs_coupling/v4_o2a/lbl_ref"
LINES = "/home/harmel/Dropbox/Dropbox/work/VRTC/sciatran/DATA_BASES/HITRAN_2012/linepars_o2.dat"
NU_RANGE = (12953., 13210.)          # cm-1, O2-A window of the ppld table (757-772 nm)
CHUNKS = range(12950, 13250, 50)     # 1 cm-1 files coef_<GAS>_<hi>_<lo>_1cmm1 covering NU_RANGE
GASES_OFF = ["CH4", "CO", "CO2", "H2O", "N2O", "NO2", "O3"]
WING_CUT = 50.                       # line wings cut at 50 half-widths (HAPI OmegaWingHW of make_o2_lbl_lut.py)
MASS = {1: 31.98983, 2: 33.99408, 3: 32.99405}   # amu, HITRAN isotopologues 66, 68, 67
OPTS = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv if a.startswith("--"))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]


# ----------------------------------------------------------------------------------------------------------------
# line-by-line cross sections and CKD files
# ----------------------------------------------------------------------------------------------------------------
def read_lines():
    rows = []
    with open(LINES) as f:
        for line in f:
            if line.startswith(";"):
                continue
            t = line.replace("'O2  '", "").split()
            rows.append([float(x) for x in t[:8]])
    a = np.array(rows)
    keep = (a[:, 1] > NU_RANGE[0] - 30) & (a[:, 1] < NU_RANGE[1] + 30)
    iso, nu0, S, gair, E, n, gself, shift = a[keep].T
    return dict(iso=iso.astype(int), nu0=nu0, S=S, gair=gair, E=E, n=n, shift=shift)


def cross_section(nu, P, T, L):
    """O2 absorption cross section (cm2 molec-1) at the wavenumbers nu (cm-1), pressure P (hPa), temperature T (K):
    Voigt profiles (air broadening and shift, no line mixing), HITRAN intensities scaled to T (rotational partition
    function ~ T)."""
    from scipy.special import voigt_profile
    c2, Tref = 1.4387770, 296.
    S = L["S"] * (Tref / T) * np.exp(-c2 * L["E"] * (1 / T - 1 / Tref)) \
        * (1 - np.exp(-c2 * L["nu0"] / T)) / (1 - np.exp(-c2 * L["nu0"] / Tref))
    p = P / 1013.25
    gam = L["gair"] * p * (Tref / T) ** L["n"]
    m = np.array([MASS[i] for i in L["iso"]]) * 1.66053907e-27
    sig = L["nu0"] / 2.99792458e8 * np.sqrt(1.380649e-23 * T / m)
    centre = L["nu0"] + L["shift"] * p
    d = np.asarray(nu)[:, None] - centre
    v = voigt_profile(d, sig, gam)
    # wings cut at WING_CUT half-widths, as HAPI for the o2_lbl_tau.nc table of radcalnet_oc (without the cut,
    # the far Lorentz wings of the 466 lines add ~7 % of optical thickness)
    v[np.abs(d) > WING_CUT * np.maximum(gam, sig * np.sqrt(2 * np.log(2)))] = 0.
    return (S * v).sum(1)


def parse_ckd(path):
    """Header lines and blocks [(first line, other lines)] of a CKD ASCII file."""
    with open(path) as f:
        lines = f.readlines()
    def is_block(t):
        try:
            return len(t) == 6 and "." in t[0] and float(t[0]) < 10 and t[5].isdigit()
        except ValueError:
            return False

    blocks, header, cur = [], [], None
    for ln in lines:
        if is_block(ln.split()):
            cur = [ln, []]
            blocks.append(cur)
        elif cur is None:
            header.append(ln)
        else:
            cur[1].append(ln)
    return header, blocks


def grid_of(header):
    """T and P grids of a CKD file (non-H2O)."""
    def numeric(t):
        try:
            [float(x) for x in t]
            return True
        except ValueError:
            return False
    i = next(k for k, ln in enumerate(header) if len(ln.split()) == 3 and numeric(ln.split()))
    T = np.array(header[i + 2].split(), float)
    P = np.array(header[i + 4].split(), float)
    return T, P


def make_files():
    rng = np.random.default_rng(20261121)
    n_samples = int(OPTS.get("samples", 4))
    L = read_lines()
    src = os.path.join(SOS, "fic", "COEFF_CKD", "1cmm1")
    sets = ["ckd"] + [f"s{k}" for k in range(n_samples)]
    samples = []
    for s in sets:
        root = os.path.join(OUT, "roots", s)
        ckd_dir = os.path.join(root, "fic", "COEFF_CKD", "1cmm1")
        os.makedirs(ckd_dir, exist_ok=True)
        for e in os.listdir(SOS):
            if e != "fic" and not os.path.lexists(os.path.join(root, e)):
                os.symlink(os.path.join(SOS, e), os.path.join(root, e))
        for e in os.listdir(os.path.join(SOS, "fic")):
            if e != "COEFF_CKD" and not os.path.lexists(os.path.join(root, "fic", e)):
                os.symlink(os.path.join(SOS, "fic", e), os.path.join(root, "fic", e))
        for e in os.listdir(os.path.join(SOS, "fic", "COEFF_CKD")):
            if e != "1cmm1" and not os.path.lexists(os.path.join(root, "fic", "COEFF_CKD", e)):
                os.symlink(os.path.join(SOS, "fic", "COEFF_CKD", e), os.path.join(root, "fic", "COEFF_CKD", e))
        modified = set()
        for lo in CHUNKS:
            hi = lo + 50
            for g in GASES_OFF:
                name = f"coef_{g}_{hi}_{lo}_1cmm1"
                header, blocks = parse_ckd(os.path.join(src, name))
                with open(os.path.join(ckd_dir, name), "w") as f:
                    f.writelines(header)
                    for first, _ in blocks:
                        t = first.split()
                        f.write(" ".join(t[:5]) + "  0\n")
                modified.add(name)
            name = f"coef_O2_{hi}_{lo}_1cmm1"
            header, blocks = parse_ckd(os.path.join(src, name))
            if s == "ckd":
                with open(os.path.join(ckd_dir, name), "w") as f:
                    f.writelines(header)
                    for first, rest in blocks:
                        f.write(first)
                        f.writelines(rest)
            else:
                T, P = grid_of(header)
                firsts = [b[0].split() for b in blocks]
                nmax = np.array([float(t[3]) for t in firsts])
                nmin = np.array([float(t[4]) for t in firsts])
                nu = nmin + rng.uniform(0.02, 0.98, len(nmin)) * (nmax - nmin)
                K = np.array([[cross_section(nu, p, tt, L) for tt in T] for p in P])   # (P, T, nu)
                with open(os.path.join(ckd_dir, name), "w") as f:
                    f.writelines(header)
                    for j in range(len(nu)):
                        f.write(f"{1e4 / nmax[j]:.8f} {1e4 / nu[j]:.8f} {1e4 / nmin[j]:.8f} {nmax[j]:.1f} "
                                f"{nmin[j]:.1f}  1\n1.0\n")
                        for ip in range(len(P)):
                            f.write(f" 1 {ip + 1:3d} " + " ".join(f"{max(v, 1e-40):.10e}" for v in K[ip, :, j])
                                    + "\n")
                        samples.append((s, nmin[j], nmax[j], nu[j]))
            modified.add(name)
        for e in os.listdir(src):
            if e not in modified and not os.path.lexists(os.path.join(ckd_dir, e)):
                os.symlink(os.path.join(src, e), os.path.join(ckd_dir, e))
        print(s, "done", flush=True)
    import pandas as pd
    pd.DataFrame(samples, columns=["set", "nu_min", "nu_max", "nu"]).to_csv(os.path.join(OUT, "samples.csv"),
                                                                           index=False)


# ----------------------------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------------------------
def load_make_luts(n_workers):
    """Functions and settings of make_luts.py (v4), without running it."""
    src = open(os.path.join(HERE, "make_luts.py")).read().split('if SET == "v4":')[0]
    argv = sys.argv
    sys.argv = ["make_luts.py", "v4", OUT, str(n_workers), "--run"]
    ns = {"__name__": "make_luts", "__file__": os.path.join(HERE, "make_luts.py")}
    try:
        exec(compile(src, "make_luts.py", "exec"), ns)
    finally:
        sys.argv = argv
    return ns


def run():
    import pandas as pd
    import xarray as xr
    n_workers = int(OPTS.get("n-workers", 12))
    ml = load_make_luts(n_workers)
    from sosabs.lut import write_netcdf
    samples = pd.read_csv(os.path.join(OUT, "samples.csv"))
    sel = (samples.nu_min >= NU_RANGE[0] - 1) & (samples.nu_max <= NU_RANGE[1] + 1)
    samples = samples[sel]
    centres = samples[samples.set == "s0"][["nu_min", "nu_max"]].mean(1).values
    sets = {"ckd": np.sort(1e4 / centres)}
    for s in sorted(samples.set.unique()):
        sets[s] = np.sort(1e4 / samples[samples.set == s].nu.values)
    if "only" in OPTS:
        sets = {k: v for k, v in sets.items() if k in OPTS["only"].split(",") + ["noabs"]}
    if "n-wl" in OPTS:
        sets = {k: v[:int(OPTS["n-wl"])] for k, v in sets.items()}
    conf = {k: v for k, v in ml["V4_NOMINAL"].items() if k != "aerosol"}
    fixed = {**ml["BASE_V4"], "-AP.SpectralResol": 1.0, "-SOS.AbsModeCKD": 1, **conf, "-AP.H2O": 2.0}
    grid = {"-ANG.Thetas": [20., 45., 70.], "-SURF.Alb": [0.0]}
    tgfixed = {**fixed, "-ANG.Rad.NbGauss": 8, "-SOS.View.Dphi": 30, "-SOS.IGmax": 1, "-SURF.Type": 0,
               "-SURF.Alb": 0.0}
    tggrid = {"-ANG.Thetas": list(ml["SZA_TG_V4"])}
    extdir = os.path.join(ml["CACHE"], "extdata_v4")
    os.makedirs(extdir, exist_ok=True)
    aot_axis = [0.05, 0.2]
    todo = [(s, wls, f, fx, gr) for s, wls in sets.items()
            for f, fx, gr in (("abs.nc", fixed, grid), ("tgdir_abs.nc", tgfixed, tggrid))]
    if "only" not in OPTS or "noabs" in OPTS["only"].split(","):
        wl_noabs = np.arange(0.7545, 0.7745, 0.0005)
        todo += [("noabs", wl_noabs, "noabs.nc", {**fixed, "-AP.AbsProfile.Type": 7}, grid),
                 ("noabs", wl_noabs, "tgdir_noabs.nc", {**tgfixed, "-AP.AbsProfile.Type": 7}, tggrid)]
    for s, wls, fname, fx0, gr in todo:
        out = os.path.join(OUT, s, fname)
        if os.path.exists(out):
            print("exists:", out)
            continue
        root = os.path.join(OUT, "roots", "ckd" if s == "noabs" else s)
        os.environ["SOS_ABS_ROOT"] = root
        t0 = time.time()
        parts = []
        for wl in wls:
            for aot in (aot_axis if not fname.startswith("tgdir") else [0.05]):
                wl, aot = float(wl), float(aot)
                fx = {**fx0, "-SOS_Main.Wa": wl, **ml["aerosol_params"]("COAV_rh70", wl, aot, extdir)}
                coords = {"wl": [wl], **({"aot550": [aot]} if not fname.startswith("tgdir") else {})}
                parts.append((coords, gr, fx))
        ds = ml["run_parts"](parts, attrs={"set": s, "sos_root": root})
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_netcdf(ds, out)
        print(f"{s} {fname}: {len(wls)} wavelengths in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------
BANDS = {"PRISMA": (764.9, 10.), "EnMAP": (763.7, 8.8)}


def analyse():
    import pandas as pd
    import xarray as xr
    sys.path.insert(0, HERE)
    import coupling as cp
    noabs = cp.open_lut(os.path.join(OUT, "noabs", "noabs.nc"))
    tgn = cp.open_lut(os.path.join(OUT, "noabs", "tgdir_noabs.nc")).flux_dir_down
    sets = sorted(d for d in os.listdir(OUT) if os.path.exists(os.path.join(OUT, d, "abs.nc")))
    rows, spectra = [], {}
    for s in sets:
        ab = cp.open_lut(os.path.join(OUT, s, "abs.nc"))
        tga = cp.open_lut(os.path.join(OUT, s, "tgdir_abs.nc")).flux_dir_down
        wl = ab.wl.values
        keep = (ab.gauss_weight > 0) & (ab.vza <= 40)
        ab = ab.isel(vza=np.flatnonzero(keep.values))
        nb = noabs.isel(vza=np.flatnonzero(keep.values)).interp(wl=wl)
        tg = (tga / tgn.interp(wl=wl)).squeeze(drop=True)
        tg = tg.assign_coords(m=1 / np.cos(np.radians(tg.sza))).swap_dims(sza="m").drop_vars("sza")
        R = {k: (d.I_up / np.cos(np.radians(d.sza))).sel(albedo=0) for k, d in (("abs", ab), ("noabs", nb))}
        R = {k: v.rename(aot550="aot_ref") if "aot550" in v.dims else v for k, v in R.items()}
        geo = cp.geometry(ab)
        Tdir = cp.tg_of_m(tg, geo.m)
        gang = np.degrees(np.arccos((geo.mu0 * geo.muv + np.sqrt(1 - geo.mu0 ** 2) * np.sqrt(1 - geo.muv ** 2)
                                     * np.cos(np.radians(ab.azi))).clip(-1, 1)))
        lam = wl                                  # nm (open_lut)
        for b, (c, fw) in BANDS.items():
            # band integral in wavelength: samples of 1 cm-1 intervals weighted by lambda^2 (d lambda = lambda^2 d nu)
            g = xr.DataArray(np.exp(-4 * np.log(2) * ((lam - c) / fw) ** 2) * lam ** 2, coords={"wl": wl})
            band = lambda x: (x * g).sum("wl") / g.sum()
            ds = xr.Dataset({"Tatm": band(R["abs"]) / band(R["noabs"]),
                             "Tdir": band(Tdir * R["noabs"]) / band(R["noabs"]),
                             "m": geo.m, "glint_angle": gang})
            df = ds.to_dataframe().reset_index()
            df = df[(df.azi < 360) & (df.m < 2.6) & (df.glint_angle > 30)]
            df["f_path"] = df.Tatm / df.Tdir
            df["set"], df["band"] = s, b
            rows.append(df)
    df = pd.concat(rows)
    df["kind"] = np.where(df.set == "ckd", "ckd", "lbl")
    key = ["band", "aot_ref", "sza", "vza", "azi"]
    lbl = df[df.kind == "lbl"].groupby(key)[["Tatm", "Tdir"]].mean()   # mean of the sample sets (equal weights)
    lbl["f_path"] = lbl.Tatm / lbl.Tdir
    ckd = df[df.kind == "ckd"].set_index(key)[["Tatm", "Tdir", "f_path"]]
    comp = lbl.join(ckd, lsuffix="_lbl", rsuffix="_ckd")
    comp["df"] = comp.f_path_lbl - comp.f_path_ckd
    pd.set_option("display.width", 200)
    print(comp.groupby(["band", "aot_ref"]).median().round(4).to_string())
    print("\nspread of f_path between the sample sets (median per band, aot):")
    print(df[df.kind == "lbl"].groupby(["band", "aot_ref", "set"]).f_path.median().unstack("set").round(4).to_string())
    comp.reset_index().to_csv(os.path.join(OUT, "fpath.csv"), index=False)


if __name__ == "__main__":
    if not ARGS or ARGS[0] not in ("files", "run", "analyse"):
        sys.exit(__doc__)
    {"files": make_files, "run": run, "analyse": analyse}[ARGS[0]]()
