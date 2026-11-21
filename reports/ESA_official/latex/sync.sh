#!/bin/bash
# Synchronise the LaTeX documents with their Overleaf projects (git remote "overleaf" of each folder):
# local changes are committed, the Overleaf changes pulled (rebase), then everything pushed.
# Usage: bash sync.sh [message]        (from anywhere; all the linked folders)
#        bash sync.sh "message" VEIT-Q  (one document: VEIT-Q, VEIT-U, GABU-Q, GABU-U)
set -e
cd "$(dirname "$0")"
declare -A DOCS=([VEIT-Q]=QA4EO-WGCV-RadCalNet-Venise-Q-v1 [VEIT-U]=QA4EO-WGCV-RadCalNet-Venise-U-v1
                 [GABU-Q]=QA4EO-WGCV-RadCalNet-Galata-Q-v1 [GABU-U]=QA4EO-WGCV-RadCalNet-Galata-U-v1)
msg=${1:-"Local edits $(date +%F)"}
codes=${2:-"VEIT-Q VEIT-U GABU-Q GABU-U"}
for code in $codes; do
  d=${DOCS[$code]}
  ( cd "$d"
    git add -A
    git diff --cached --quiet || git commit -q -m "$msg"
    git pull -q --rebase overleaf main
    git push -q overleaf main
    echo "$code ($d): $(git log -1 --format='%h %s' | cut -c1-70) [$(git status -sb | head -1 | sed 's/## //')]" )
done
