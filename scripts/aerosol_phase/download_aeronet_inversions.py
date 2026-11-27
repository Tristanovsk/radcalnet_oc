'''
Download the AERONET Version 3 almucantar inversions of a site (all points, Level 1.5): phase function (PFN),
extinction AOD (AOD) and single scattering albedo (SSA), into ``<AEROSOL_PHASE_DIR>/aeronet``.

Level 1.5 is used because the Level 2.0 files leave the low-AOD retrievals as fill values: the retrievals of
Level 2 quality without the AOD(440) > 0.4 threshold are selected later with the column
``If_Retrieval_is_L2(without_L2_0.4_AOD_440_threshold)``.

Usage:
    python download_aeronet_inversions.py [--site AAOT] [--start 2002] [--end 2026]
'''
import argparse
import os
import urllib.request

from phase_common import AERONET_DIR

URL = ('https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?site={site}&year={y0}&month=1&day=1'
       '&year2={y1}&month2=12&day2=31&product={product}&AVG=10&ALM15=1&if_no_html=1')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='AAOT')
    p.add_argument('--start', type=int, default=2002)
    p.add_argument('--end', type=int, default=2026)
    a = p.parse_args()
    os.makedirs(AERONET_DIR, exist_ok=True)
    for product in ['PFN', 'AOD', 'SSA']:
        out = os.path.join(AERONET_DIR, f'{a.site}_{product}_ALM15.txt')
        url = URL.format(site=a.site, y0=a.start, y1=a.end, product=product)
        with urllib.request.urlopen(url, timeout=600) as r, open(out, 'wb') as f:
            f.write(r.read())
        with open(out) as f:
            n = sum(1 for _ in f) - 7
        print(f'{out}: {n} retrievals')
