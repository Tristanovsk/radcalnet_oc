"""
Build the line-by-line O2 absorption table of radcalnet_oc (radcalnet_oc/data/lut/gases/o2_lbl_tau.nc).

Vertical absorption optical thickness of O2 in the O2-B (682.6-701.8 nm) and O2-A (754.7-775.2 nm) bands:

- HITRAN lines of 16O2, 16O18O and 16O17O (global isotopologue ids 36, 37, 38), fetched with HAPI
  (``pip install hitran-api``; internet access needed the first time);
- Voigt profiles with air broadening (HAPI ``absorptionCoefficient_Voigt``, wings of 50 half-widths), without
  line mixing nor O2-O2 collision-induced absorption;
- AFGL midlatitude summer profile (libRadtran ``afglms.dat``, 0-60 km, 59 layers), surface pressure 1013 hPa,
  O2 column 4.51e24 molec cm-2;
- wavenumber step 0.004 cm-1 (about a third of the Doppler half-width), vacuum wavelengths.

radcalnet_oc scales the optical thickness with the surface pressure and averages exp(-m tau) in the cells of
the gaseous look-up table (see GaseousTransmittance, o2_model='lbl').

Usage::

    python scripts/make_o2_lbl_lut.py --afgl /path/to/libRadtran/data/atmmod/afglms.dat [--cache hitran_cache]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

BANDS = {'O2-B': (14250., 14650.), 'O2-A': (12900., 13250.)}      # cm-1
OUT = Path(__file__).resolve().parents[1] / 'radcalnet_oc' / 'data' / 'lut' / 'gases' / 'o2_lbl_tau.nc'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--afgl', default='/usr/local/share/libRadtran/data/atmmod/afglms.dat',
                        help='AFGL atmosphere file of libRadtran (z, p, T, air, o3, o2...)')
    parser.add_argument('--cache', default='hitran_cache', help='folder of the HITRAN line lists')
    parser.add_argument('--step', type=float, default=0.004, help='wavenumber step (cm-1)')
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()

    import hapi
    hapi.db_begin(args.cache)
    for band, (lo, hi) in BANDS.items():
        if band not in hapi.tableList():
            hapi.fetch_by_ids(band, [36, 37, 38], lo - 5, hi + 5)

    z, p, T, _, _, o2 = np.loadtxt(args.afgl, comments='#', usecols=range(6)).T
    order = np.argsort(z)
    z, p, T, o2 = z[order], p[order], T[order], o2[order]
    keep = z <= 60
    z, p, T, o2 = z[keep], p[keep], T[keep], o2[keep]
    column = np.sqrt(o2[1:] * o2[:-1]) * (z[1:] - z[:-1]) * 1e5        # molec cm-2 per layer
    p_layer, t_layer = np.sqrt(p[1:] * p[:-1]) / 1013.25, 0.5 * (T[1:] + T[:-1])

    out = []
    for band, (lo, hi) in BANDS.items():
        t0 = time.time()
        tau = 0.
        for i in range(len(column)):
            nu, xs = hapi.absorptionCoefficient_Voigt(SourceTables=band, OmegaRange=[lo, hi], OmegaStep=args.step,
                                                      Environment={'p': p_layer[i], 'T': t_layer[i]},
                                                      HITRAN_units=True, OmegaWingHW=50., Diluent={'air': 1.0})
            tau = tau + xs * column[i]
        out.append(xr.DataArray(tau[::-1], dims='wl', coords={'wl': 1e7 / nu[::-1]}))
        print(f'{band}: {time.time() - t0:.0f} s')

    tau = xr.concat(out, 'wl').rename('tau_o2')
    tau['wl'].attrs = {'units': 'nm', 'description': 'vacuum wavelength'}
    tau.attrs = {'description': 'vertical absorption optical thickness of O2 in the O2-B and O2-A bands, '
                                'line by line (HITRAN, HAPI Voigt profiles, no line mixing, no collision-induced '
                                'absorption), AFGL midlatitude summer profile',
                 'surface_pressure': 1013., 'surface_pressure_units': 'hPa', 'o2_column': float(column.sum()),
                 'o2_column_units': 'molec cm-2', 'wavenumber_step': args.step,
                 'bands': 'O2-B: {:.3f}-{:.3f} nm; O2-A: {:.3f}-{:.3f} nm'.format(
                     *[1e7 / v for lo, hi in BANDS.values() for v in (hi, lo)]),
                 'created_by': 'scripts/make_o2_lbl_lut.py'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tau.to_netcdf(args.output, encoding={'tau_o2': {'zlib': True, 'complevel': 9, 'dtype': 'float32'}})
    print(f'saved {args.output} ({tau.size} points)')


if __name__ == '__main__':
    main()
