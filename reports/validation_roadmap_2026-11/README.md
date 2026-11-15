# Validation, performance and roadmap (November 2026)

LaTeX source of the report *RadCalNet-OC: validation, performance and roadmap*, edited on Overleaf
(project `6afa1eff4f8b6aee6f8d4f83`, which supports git: `git clone https://git.overleaf.com/6afa1eff4f8b6aee6f8d4f83`).
This folder is a copy; sync it from Overleaf after editing there.

Build: `latexmk -pdf main.tex`

Figures: regenerated from the Sentinel-2 matchup files with

    python scripts/s2_toa_matchup/matchup_analysis.py figures --fig-dir reports/validation_roadmap_2026-11/figures

and the tables / numbers of the text with the `stats` and `drivers` commands of the same script.
