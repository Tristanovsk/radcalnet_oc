"""OSOAA runs of the LUT configuration at wavelengths between the nodes of the TOA LUT.

The runs of /data/vrtc/lut/atmo_full only exist up to 1.0 um and at 1.6, 2.2, 2.4, 2.5 um: nothing checks the
interpolation of the TOA LUT between 1.0 and 1.6 um. The parameters are copied from the run of the 1.0 um node,
with the wavelength-dependent ones set as in the LUT runs: Rayleigh optical thickness of Bodhaine et al. (1999)
scaled to 1015.2 hPa (AP.MOT), depolarisation of the same table (DPOL.AIR), scattering matrix of the aerosol model
at the wavelength (mopsmap, 50 nm grid), refractive index of the water interpolated linearly between the runs at 1.0
and 1.6 um (SEA.Ind; its effect on the path reflectance is negligible). Uses the OSOAA executable read-only, as
scripts/rayleigh_pressure/run_pressure.py.

Usage: python run_offnode.py <wavelengths um> <aot:model,...> [n processes] [output folder]
    python run_offnode.py 1.15,1.25,1.4 0.0000:COAV_rh70,0.1:COAV_rh70,0.1:MACL_rh70,0.35:DESE_rh70 4
The first run of each wavelength computes its surface matrices: the cases are run wavelength by wavelength in
parallel, the first aerosol case of each wavelength first.
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rayleigh_pressure'))
from run_pressure import EXE, EXE_DIR, KEYS, FILES, P_LUT, lut_file  # noqa: E402

from radcalnet_oc.lut import rayleigh_file  # noqa: E402

OUT = '/data/vrtc/lut/osoaa_offnode'
SZAS = [20, 40, 60]
WIND = '2'
SCATMAT = '/DATA/git/vrtc/mopsmap_tbx/scatmat/ascii/scat_mat_opac_{model}_wl{wl:.1f}.csv'


def rayleigh(wl_um):
    '''Rayleigh optical thickness at P_LUT and depolarisation factor (Bodhaine et al., 1999).'''
    d = pd.read_csv(rayleigh_file, skiprows=16, sep=' ', header=None, names=['wl', 'rot', 'dpol']).set_index('wl')
    row = d.loc[round(wl_um * 1000)]
    return row.rot * P_LUT / 1013.25, row.dpol


def attributes(aot, model, wl):
    '''Parameters of the run: those of the 1.0 um node with the wavelength-dependent ones at `wl`.'''
    at = xr.open_dataset(lut_file(aot, model, WIND, 1.0)).attrs
    n10 = float(at['SEA.Ind'])
    n16 = float(xr.open_dataset(lut_file(aot, model, WIND, 1.6)).attrs['SEA.Ind'])
    args = {k: at[k] for k in KEYS}
    mot, dpol = rayleigh(wl)
    scatmat = SCATMAT.format(model=model, wl=wl * 1000)
    if not os.path.exists(scatmat):
        raise FileNotFoundError(scatmat)
    args.update({'OSOAA.Wa': f'{wl:.3f}', 'AP.MOT': f'{mot:.14f}', 'DPOL.AIR': f'{dpol:.14f}',
                 'AER.ExtData': scatmat, 'SEA.Ind': f'{np.interp(wl, [1.0, 1.6], [n10, n16]):.5f}'})
    return args


def run(case):
    aot, model, wl, out = case
    name = f'osoaa_atmo_full_aot{aot}_aer{model}_HR8.0_HA2.0_ws{WIND}_wl{wl:0.3f}'
    ofile = f'{out}/out/{name}.nc'
    if os.path.exists(ofile):
        return ofile, 0.
    args = attributes(aot, model, wl)
    root = f'{out}/work/{name}'
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for d in ['MIE_AER', 'MIE_HYD', 'SURF_MATR']:
        os.makedirs(f'{out}/db/{d}', exist_ok=True)
    args.update({'OSOAA.ResRoot': root, 'AER.DirMie': f'{out}/db/MIE_AER', 'HYD.DirMie': f'{out}/db/MIE_HYD',
                 'SEA.Dir': f'{out}/db/SURF_MATR', **FILES})
    t0 = time.time()
    adv = f'{root}/Advanced_outputs'
    for i, sza in enumerate(SZAS):
        cmd = [EXE] + [x for k, v in {**args, 'ANG.Thetas': str(sza)}.items() for x in (f'--{k}', str(v))]
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'{name} sza {sza}: {r.stderr[-500:]}')
        line = ' '.join(cmd)
        with open(f'{adv}/stokes_profile.txt', 'a') as fs, open(f'{adv}/fluxes_profile.txt', 'a') as ff:
            up = open(f'{adv}/RESLUM_Advanced_UP.txt').read().splitlines(True)
            down = open(f'{adv}/flux_down.txt').read().splitlines(True)
            if i == 0:
                fs.write(line + '\n'); fs.writelines(up)
                ff.write(line + '\n'); ff.writelines(down)
            else:
                fs.writelines(up[35:]); ff.writelines(down[31:])
    os.makedirs(f'{out}/out', exist_ok=True)
    r = subprocess.run([sys.executable, f'{EXE_DIR}/lut_package/lut_generator.py', root, ofile], cwd=EXE_DIR,
                       capture_output=True, text=True)
    # with aerosols the converter may stop on the aerosol metadata after writing the radiances (see run_pressure)
    if not os.path.exists(ofile):
        raise RuntimeError(f'{name} conversion: {r.stderr[-800:]}')
    shutil.rmtree(root)
    return ofile, time.time() - t0


if __name__ == '__main__':
    wls = [float(w) for w in sys.argv[1].split(',')]
    aero = [tuple(a.split(':')) for a in sys.argv[2].split(',')]
    njobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = sys.argv[4] if len(sys.argv) > 4 else OUT
    # empty surface matrix files left by an interrupted run: OSOAA would read them instead of recomputing them
    surf = f'{out}/db/SURF_MATR'
    for f in (os.listdir(surf) if os.path.isdir(surf) else []):
        if os.path.getsize(f'{surf}/{f}') == 0:
            print(f'removed empty surface matrix {f}', flush=True)
            os.remove(f'{surf}/{f}')
    first = [(*aero[0], wl, out) for wl in wls]  # surface matrices of each wavelength
    rest = [(*a, wl, out) for wl in wls for a in aero[1:]]
    for cases in (first, rest):
        with ProcessPoolExecutor(njobs) as ex:
            for f, dt in ex.map(run, cases):
                print(f'{os.path.basename(f)} {dt:.0f} s', flush=True)
