"""Extract the figures of an executed notebook, in their order of appearance: figure_1.png, figure_2.png...

Usage: python extract_figures.py notebook.ipynb [out_dir]
"""
import base64
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit(__doc__)
out = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(out, exist_ok=True)
nb = json.load(open(sys.argv[1]))
n = 0
for cell in nb['cells']:
    for o in cell.get('outputs', []):
        if 'image/png' in o.get('data', {}):
            n += 1
            with open(os.path.join(out, f'figure_{n}.png'), 'wb') as f:
                f.write(base64.b64decode(o['data']['image/png']))
print(n, 'figures written to', out)
