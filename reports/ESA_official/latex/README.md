# RadCalNet site documents in LaTeX, synchronised with Overleaf

The Word documents of `..` converted to LaTeX (`convert.sh`: pandoc, then `clean.py`), one folder per document
(`main.tex`, `media/`). Site codes: **VEIT** = Venise (AERONET-OC AAOT), **GABU** = Galata, **CORO** = Section-7
Platform.

| code | folder | Overleaf project |
|---|---|---|
| VEIT-Q | `QA4EO-WGCV-RadCalNet-Venise-Q-v1` | https://www.overleaf.com/project/6afdd2e1bdefd58acae75c9d |
| VEIT-U | `QA4EO-WGCV-RadCalNet-Venise-U-v1` | https://www.overleaf.com/project/6afdd309590f14297f86e4c4 |
| GABU-Q | `QA4EO-WGCV-RadCalNet-Galata-Q-v1` | https://www.overleaf.com/project/6afdd357fe4dcbdcb75d5fed |
| GABU-U | `QA4EO-WGCV-RadCalNet-Galata-U-v1` | https://www.overleaf.com/project/6afdd370c8062c3a8df0ac84 |
| CORO-U | `QA4EO-WGCV-RadCalNet-CORO-U-v1` | https://www.overleaf.com/project/6b02ba23392fcfa94d0aa6ac (written from VEIT-U revision 3) |
| CORO-Q | `QA4EO-WGCV-RadCalNet-CORO-Q-v1` | https://www.overleaf.com/project/6b02b9f77e720d9757d0fcfa (in preparation, from GABU-Q) |

The figures of the uncertainty statements (`media/v1.3`) are made by `scripts/radcalnet_products/statement_figures.py`
(products, budget, matchups with the ROI and the sunglint flags) and `scripts/radcalnet_products/site_rgb.py`
(Sentinel-2 image at 10 m with the ROI, also in section 2.3 of the Q documents); see their docstrings for the
commands of VEIT, GABU and CORO.

The ACTION-TN (`ACTION-TN-101-MAGv2.0_CCN4_D2_20250818`) is converted but not linked to Overleaf.

## Synchronisation

Each of the linked folders is a git repository whose remote `overleaf` is its Overleaf project (Overleaf git
integration; the token is kept by the git credential helper of this machine).

```bash
bash sync.sh                         # all documents: commit local edits, pull Overleaf edits, push
bash sync.sh "Fix table 3-1" GABU-Q  # one document, with a commit message
```

or by hand in a folder: `git pull --rebase overleaf main`, `git push overleaf main`. Overleaf commits the edits made
in its editor when they are pulled; edit the same paragraph on one side at a time to avoid conflicts (a conflict is
solved as in any git repository, then `git rebase --continue`).

**Do not rerun `convert.sh` on a linked document once it is edited**: it regenerates `main.tex` from the Word file
and would overwrite the edits at the next sync (the folders are emptied except `.git` and `.gitignore`). To recover a
version, use the git history of the folder or the history of the Overleaf project.
