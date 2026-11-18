"""OSOAA runs of the LUT configuration at several pressures (validation of the Rayleigh pressure scaling).

Parameters copied from the attributes of the LUT runs (/data/vrtc/lut/atmo_full); only AP.MOT is scaled by
P / 1015.2 hPa (pressure of the LUT runs). The executable and converter of OSOAA_profile are used read-only;
all the outputs, working files and surface matrices are written under this script's folder.
"""
import os, sys, subprocess, shutil, itertools, time
import xarray as xr
from concurrent.futures import ProcessPoolExecutor

EXE = '/DATA/VRTC/OSOAA_profile/exe/OSOAA_MAIN_profile.exe'
EXE_DIR = '/DATA/VRTC/OSOAA_profile/exe'
LUT_DIR = '/data/vrtc/lut/atmo_full'
HERE = os.path.dirname(os.path.abspath(__file__))
P_LUT = 1015.2
KEYS = ['OSOAA.Wa', 'DPOL.AIR', 'ANG.Rad.NbGauss', 'ANG.Mie.NbGauss', 'AER.Tronca', 'AER.Waref', 'AER.AOTref',
        'AER.Model', 'AER.ExtData', 'AER.ExtDataWaRef', 'HYD.Model', 'PHYTO.JD.slope', 'PHYTO.JD.rmin',
        'PHYTO.JD.rmax', 'PHYTO.JD.MRwa', 'PHYTO.JD.MIwa', 'PHYTO.JD.rate', 'PHYTO.LND.SM.SDradius',
        'PHYTO.LND.SM.SDvar', 'PHYTO.LND.SM.MRwa', 'PHYTO.LND.SM.MIwa', 'PHYTO.LND.SM.rate', 'PHYTO.LND.TM.SDradius',
        'PHYTO.LND.TM.SDvar', 'PHYTO.LND.TM.MRwa', 'PHYTO.LND.TM.MIwa', 'PHYTO.LND.TM.rate', 'HYD.ExtData', 'AP.HR',
        'AP.HA', 'SEA.Depth', 'PHYTO.ProfilType', 'PHYTO.Chl', 'PHYTO.GP.Chlbg', 'PHYTO.GP.Deep', 'PHYTO.GP.Width',
        'SED.Csed', 'SED.JD.slope', 'SED.JD.rmin', 'SED.JD.MRwa', 'SED.JD.MIwa', 'SED.JD.rate', 'YS.Abs440', 'YS.Swa',
        'DET.Abs440', 'DET.Swa', 'SEA.Ind', 'SEA.Wind', 'SEA.SurfAlb', 'SEA.BotType', 'SEA.BotAlb', 'SOS.IGmax']
FILES = {'OSOAA.Log': 'Trace_Main.Log', 'ANG.Rad.ResFile': 'LUM_UsedAngles.txt', 'ANG.Mie.ResFile': 'AER_UsedAngles.txt',
         'AER.ResFile': 'AER_Granu.txt', 'ANG.Log': 'Trace_Angles.Log', 'AER.MieLog': 'Trace_AerMie.Log',
         'AER.Log': 'Trace_Aerosols.Log', 'HYD.Log': 'Trace_hydrosols.Log', 'PHYTO.ResFile': 'Phyto_Granu.txt',
         'MLP.ResFile': 'Sediments_Granu.txt', 'HYD.MieLog': 'Trace_HydMie.Log', 'PROFILE_ATM.ResFile': 'PROFILE_ATM.txt',
         'PROFILE_SEA.ResFile': 'PROFILE_SEA.txt', 'PROFILE.Log': 'Trace_Profils.Log', 'SEA.Log': 'Trace_Surface.Log',
         'OSOAA.ResFile.vsVZA': 'RESLUM_vsVZA.txt', 'OSOAA.ResFile.vsZ': 'RESLUM_vsZ.txt',
         'OSOAA.ResFile.Adv.Up': 'RESLUM_Advanced_UP.txt', 'OSOAA.ResFile.Adv.Down': 'flux_down.txt',
         'SOS.Log': 'Trace_SOS.Log', 'SOS.ResFile.Bin': 'RESLUM.bin'}


def lut_file(aot, model, wind, wl):
    return f'{LUT_DIR}/osoaa_atmo_full_aot{aot}_aer{model}_HR8.0_HA2.0_ws{wind}_wl{wl:0.3f}.nc'


def run(case):
    aot, model, wind, wl, P, szas = case
    name = f'osoaa_atmo_full_aot{aot}_aer{model}_HR8.0_HA2.0_ws{wind}_wl{wl:0.3f}_P{P:.1f}'
    ofile = f'{HERE}/out/{name}.nc'
    if os.path.exists(ofile):
        return ofile, 0.
    attrs = xr.open_dataset(lut_file(aot, model, wind, wl)).attrs
    root = f'{HERE}/work/{name}'
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for d in ['MIE_AER', 'MIE_HYD', 'SURF_MATR']:
        os.makedirs(f'{HERE}/db/{d}', exist_ok=True)
    args = {k: attrs[k] for k in KEYS}
    args['AP.MOT'] = f'{float(attrs["AP.MOT"]) * P / P_LUT:.14f}'
    args.update({'OSOAA.ResRoot': root, 'AER.DirMie': f'{HERE}/db/MIE_AER', 'HYD.DirMie': f'{HERE}/db/MIE_HYD',
                 'SEA.Dir': f'{HERE}/db/SURF_MATR', **FILES})
    t0 = time.time()
    adv = f'{root}/Advanced_outputs'
    for i, sza in enumerate(szas):
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
    os.makedirs(f'{HERE}/out', exist_ok=True)
    r = subprocess.run([sys.executable, f'{EXE_DIR}/lut_package/lut_generator.py', root, ofile], cwd=EXE_DIR,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'{name} conversion: {r.stderr[-800:]}')
    shutil.rmtree(root)
    return ofile, time.time() - t0


if __name__ == '__main__':
    szas = [int(s) for s in sys.argv[1].split(',')]
    wls = [float(w) for w in sys.argv[2].split(',')]
    pressures = [float(p) for p in sys.argv[3].split(',')]
    aero = [a.split(':') for a in sys.argv[4].split(',')]  # aot:model
    njobs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    cases = [(aot, model, '2', wl, P, szas) for (aot, model), wl, P in itertools.product(aero, wls, pressures)]
    with ProcessPoolExecutor(njobs) as ex:
        for f, dt in ex.map(run, cases):
            print(f'{os.path.basename(f)} {dt:.0f} s', flush=True)
