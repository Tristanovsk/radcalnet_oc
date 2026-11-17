"""Check of the OPAC aerosols of the v4 set: SOS-ABS with a converted mopsmap file against the OSOAA run of the LUT.

SOS-ABS (-AER.Model 4, file written by make_luts.write_sos_extdata) and the OSOAA run of the TOA LUT of radcalnet_oc
(/data/vrtc/lut/atmo_full, black ocean, wind 2 m s-1, no gaseous absorption, Rayleigh optical thickness AP.MOT of the
run) at 0.8 um, sza 40 deg, for the Rayleigh atmosphere (AOT 0) and COAV_rh70 at AOT(550) 0.1. The two codes differ in
their Rayleigh and surface treatments; the aerosol contribution R(0.1) - R(0) is compared, where an error in the
reading of the phase matrix (angle order, columns, normalisation) would show. Controls: the same file with the angles
from 0 to 180 deg (order of mopsmap), and the scalar run of the v4 nominal set.

Usage: python test_extdata.py [model] [wl_um] [sza]
"""
import os
import sys
import tempfile

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'COAV_rh70'
WL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
SZA = float(sys.argv[3]) if len(sys.argv) > 3 else 40.
VZAS, AZIS = [0., 10., 20., 30., 40.], [0., 45., 90., 135., 180.]

# functions of make_luts.py (defined before its main part)
src = open(os.path.join(HERE, 'make_luts.py')).read()
ns = {'__file__': os.path.join(HERE, 'make_luts.py'), '__name__': 'make_luts'}
argv, sys.argv = sys.argv, ['make_luts.py', 'v4']
exec(compile(src[:src.index('if SET == "v4":\n    OUTDIR')], 'make_luts.py', 'exec'), ns)
sys.argv = argv
import sosabs  # noqa: E402  (path set by make_luts)


def osoaa(aot):
    f = f'/data/vrtc/lut/atmo_full/osoaa_atmo_full_aot{aot}_aer{MODEL}_HR8.0_HA2.0_ws2_wl{WL:0.3f}.nc'
    at = xr.open_dataset(f).attrs
    s = xr.open_dataset(f).sza.values
    with xr.open_dataset(f, group='stokes/up') as up:
        I = up.I.isel(z=-1).assign_coords(sza=s).sel(sza=SZA).interp(vza=VZAS).sel(azi=AZIS).load()
    return I.values, at


def sos(aot550, at, extfile=None, ipolar=1):
    par = {**ns['BASE_V4'], '-SOS_Main.Wa': WL, '-ANG.Thetas': SZA, '-AP.MOT': float(at['AP.MOT']),
           '-AP.HR': 8.0, '-AP.AerProfile.Type': 1, '-AP.AerHS.HA': 2.0, '-AP.AbsProfile.Type': 7,
           '-SURF.Type': 1, '-SURF.Ind': float(at['SEA.Ind']), '-SURF.Glitter.Wind': 2.0, '-SURF.Alb': 0.0,
           '-SOS.DirectSurf': 0, '-SOS.Ipolar': ipolar}
    if aot550 == 0:
        par.update({'-AER.Model': 1, '-AER.WMO.Model': 2, '-AER.Waref': WL, '-AER.AOTref': 0.0})
    else:
        ratio = ns['opac_optics'](MODEL, WL)[0]
        par.update({'-AER.Model': 4, '-AER.ExtData': extfile, '-AER.Waref': WL, '-AER.AOTref': float(aot550 * ratio)})
    work = tempfile.mkdtemp(prefix='extdata_', dir=os.path.join(ns['CACHE'], 'runs'))
    par.update({'-AER.DirMie': os.path.join(ns['CACHE'], 'MIE'), '-SURF.Dir': os.path.join(work, 'SURF')})
    os.makedirs(par['-SURF.Dir'], exist_ok=True)
    ds = sosabs.run(par, workdir=work)
    I = ds['I_up'].interp(vza=VZAS).sel(azi=AZIS)
    return I.transpose('vza', 'azi').values


os.makedirs(os.path.join(ns['CACHE'], 'runs'), exist_ok=True)
ratio, ssa, matrix = ns['opac_optics'](MODEL, WL)
tmp = tempfile.mkdtemp(prefix='extdata_files_', dir=os.path.join(ns['CACHE'], 'runs'))
desc = os.path.join(tmp, f'{MODEL}_desc.txt')
ns['write_sos_extdata'](desc, ssa, matrix)                       # converted file of make_luts (180 -> 0 deg)
asc = os.path.join(tmp, f'{MODEL}_asc.txt')
with open(asc, 'w') as f:                                          # control: angles 0 -> 180 deg
    f.write(f'EXTINCTION_COEF : {1.:.5E}\nSCATTERING_COEF : {ssa:.5E}\nNB_LINES : {len(matrix)}\n')
    f.write('Angle    F11        -F12/F11    F22/F11    F33/F11\n')
    np.savetxt(f, matrix[np.argsort(matrix[:, 0])], fmt=['%8.4f', '%.6E', '%.6E', '%.6E', '%.6E'])
print(f'{MODEL} {WL} um, sza {SZA}: AOT(wl)/AOT(550) {ratio:.4f}, ssa {ssa:.4f}, {len(matrix)} angles')

o0, at = osoaa('0.0000')
o1, _ = osoaa('0.1')
runs = {'rayleigh': sos(0, at), 'desc (make_luts)': sos(0.1, at, desc), 'asc (control)': sos(0.1, at, asc),
        'desc scalar (v4 nominal)': sos(0.1, at, desc, ipolar=0), 'rayleigh scalar': sos(0, at, ipolar=0)}


def show(label, a, b):
    rel = 100 * (a / b - 1)
    # vza 0 is outside the output angles of SOS-ABS (Gauss angles + sza): NaN, left out
    print(f'{label}: median {np.nanmedian(rel):+.2f} %, range {np.nanmin(rel):+.2f} to {np.nanmax(rel):+.2f} %')
    print('    vza x azi (%):\n' + '\n'.join('    ' + ' '.join(f'{v:+6.2f}' for v in row) for row in rel))


print(f'azimuths {AZIS}, vza {VZAS}')
show('Rayleigh, SOS vector / OSOAA', runs['rayleigh'], o0)
for label in ('desc (make_luts)', 'asc (control)'):
    show(f'aerosol term, SOS {label} / OSOAA', runs[label] - runs['rayleigh'], o1 - o0)
show('aerosol term, SOS scalar desc / OSOAA', runs['desc scalar (v4 nominal)'] - runs['rayleigh scalar'], o1 - o0)
show('total AOT 0.1, SOS vector desc / OSOAA', runs['desc (make_luts)'], o1)
