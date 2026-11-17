'''
Real-scene check of the pressure scaling of GRS: one Sentinel-2 L1C image at an AERONET-OC station.

The image is processed once with Process.execute (default settings, pressure_scaling 'rayleigh' for the
rugosity), then the AOT retrieval (default aerosol settings) and the final processing are repeated with
pressure_scaling 'linear' (GRSprocessor <= 3.1.x) and 'rayleigh'. For each, in a 3 x 3 box centred on the
station: median aot550 and Rrs, retrieved AOT of the station, CAMS surface pressure.

Usage: python grs_pressure_check.py <L1C .SAFE> <CAMS file> <lon> <lat> <output folder> [resolution]
'''
import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings('ignore')
import grs

sys.path.insert(0, os.path.join(os.path.dirname(grs.__file__), '..', 'notebook', 'settings_aerosols', 'AERONET-OC'))
from aeronet_aerosol_study import retrieve_aot  # noqa: E402

BOX = 3
WL = [443, 490, 560, 665, 705, 740, 783, 865, 1610, 2190]
AEROSOL = dict(aot_resolution=240, aot_chunk=64, min_pixel_prop=0.5, outlier_factor=1.5, aot_smoothing_blocks=3,
               fine_step=5, fine_window=None)


def main(safe, cams_file, lon, lat, odir, resolution=60):
    os.makedirs(odir, exist_ok=True)
    name = os.path.basename(safe.rstrip('/')).replace('.SAFE', '')
    out_csv = os.path.join(odir, name + '.csv')
    if os.path.exists(out_csv):
        print('already done:', out_csv)
        return
    t0 = time.time()
    process = grs.Process()
    process.execute(safe, cams_file=cams_file, resolution=resolution)
    kernel = process.kernel
    raster = kernel.prod.raster
    combination = (kernel.aero_lut.model.values == process.l2a.l2_prod.attrs['opac_model']).astype(int)

    x, y = Transformer.from_crs(4326, raster.rio.crs, always_xy=True).transform(float(lon), float(lat))
    ix = int(np.abs(raster.x.values - x).argmin())
    iy = int(np.abs(raster.y.values - y).argmin())
    half = BOX // 2
    box = dict(x=slice(max(0, ix - half), ix + half + 1), y=slice(max(0, iy - half), iy + half + 1))
    inside = (abs(raster.x.values[ix] - x) < resolution) and (abs(raster.y.values[iy] - y) < resolution)
    mask = process.l2a.l2_prod.mask.isel(**box).compute()
    water = raster.sza.isel(**box).notnull().compute()
    common = dict(image=name, grs=grs.__file__, opac_model=process.l2a.l2_prod.attrs['opac_model'],
                  station_in_image=inside, box_water=int(water.sum()), box_mask_valid=int(((mask == 0) & water).sum()),
                  pressure=float(kernel._surface_pressure.interp(x=x, y=y)),
                  cams_aod550_station=float(kernel._cams_aod550.interp(x=x, y=y)),
                  sza=float(raster.sza.isel(**box).mean()), time_execute=time.time() - t0)

    rows = []
    for scaling in ('linear', 'rayleigh'):
        t = time.time()
        kernel.pressure_scaling = scaling
        kernel.lut_preparation(aerosol_combination=combination)
        kernel.set_gas_transmittance()
        r = retrieve_aot(kernel, (x, y), **AEROSOL)
        row = dict(common, pressure_scaling=scaling)
        if r is None:
            rows.append(row)
            continue
        aot_min, aot_max = float(r['fine'].aot_ref.min()), float(r['fine'].aot_ref.max())
        kernel.lut_preparation(aerosol_combination=combination,
                               aot_refs=[0, *np.arange(aot_min, aot_max + 0.01, 0.01)])
        res = kernel.final_process(raster, r['fine'].aot_ref, chunk=512).isel(**box).compute()
        rrs = res.Rrs.where(water)
        row.update(n_blocks=r['n_blocks'], n_fallback=r['n_fallback'],
                   station_block_aot=r.get('station_block_aot', np.nan),
                   station_fallback=r.get('station_fallback', np.nan),
                   aot550=float(res.aot550.where(water).median()),
                   aot_ref_station=float(r['fine'].aot_ref.interp(x=x, y=y, method='linear')),
                   aot_ref_scene=float(r['fine'].aot_ref.median()),
                   **{f'rrs{w}': float(rrs.sel(wl=w, method='nearest').median()) for w in WL},
                   time_variant=time.time() - t)
        rows.append(row)
        print(f"{scaling:8s} P {row['pressure'] / 100:.1f} hPa aot550 {row['aot550']:.4f} "
              f"rrs443 {row['rrs443']:.5f} rrs865 {row['rrs865']:.5f} ({row['time_variant']:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'{name}: done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    main(*sys.argv[1:6], *(int(a) for a in sys.argv[6:]))
