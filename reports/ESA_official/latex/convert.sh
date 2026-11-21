#!/bin/bash
# Conversion of the ESA / RadCalNet Word documents to LaTeX projects (one folder per document: main.tex and
# media/, and <name>.zip for Overleaf: New Project > Upload Project), cleaned by clean.py. Usage: bash convert.sh  (from this folder)
set -e
cd "$(dirname "$0")"
cat > preamble.tex <<'TEX'
\usepackage[margin=2.2cm]{geometry}
\usepackage{float}
\usepackage{caption}
\floatplacement{figure}{H}
\setlength{\emergencystretch}{3em}
\usepackage{newunicodechar}
\newunicodechar{⁰}{\textdegree}
TEX
for docx in ../*.docx; do
  name=$(basename "$docx" .docx)
  # folder emptied except its git link to Overleaf (.git, .gitignore): see README.md
  mkdir -p "$name" && find "$name" -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitignore -exec rm -rf {} +
  (cd "$name" && pandoc "../$docx" -s -f docx -t latex --extract-media=. --wrap=preserve \
     -V documentclass=article -V fontsize=11pt -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
     -H ../preamble.tex -o main.tex)
  # images not readable by pdflatex (gif, emf, wmf, tif) converted to png
  for img in "$name"/media/*.{gif,emf,wmf,tif,tiff}; do
    [ -e "$img" ] || continue
    png="${img%.*}.png"
    convert "$img[0]" "$png" 2>/dev/null || soffice --headless --convert-to png --outdir "$(dirname "$img")" "$img" >/dev/null
    sed -i "s#$(basename "$img")#$(basename "$png")#g" "$name/main.tex"
    rm -f "$img"
  done
  python3 clean.py "$name"                     # sections, figures, captions, equations, title page, header/footer
  rm -f "$name/pandoc.tex"
  (cd "$name" && rm -f "../$name.zip" && zip -qr "../$name.zip" main.tex media)
  echo "$name: $(grep -c includegraphics "$name/main.tex") figures, $(grep -c '\\begin{longtable}' "$name/main.tex") tables"
done
