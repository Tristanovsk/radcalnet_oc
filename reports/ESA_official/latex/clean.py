"""Clean-up of the LaTeX converted by pandoc from the ESA / RadCalNet Word documents (convert.sh), for editing on
Overleaf: title page and running header/footer with the logos of the Word header, sections instead of the numbered
list items and hyperlinked pseudo-headings of the templates, figures and tables with captions and labels (references
to the figures kept), equations typeset instead of images, simpler tables (no minipages), widths relative to the
text width, Word table of contents replaced by \\tableofcontents.

Usage: python clean.py <name>   (folder <name> with main.tex from pandoc; the Word file ../<name>.docx)
Writes <name>/main.tex (pandoc output kept as <name>/pandoc.tex).
"""
import hashlib
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTWIDTH_IN = 6.27

# equations given as images in the uncertainty statements (md5 of the image -> LaTeX)
EQ_TOA = r'''\begin{equation}
R^{TOA}_{tot} = \frac{L^{TOA}_{tot}}{E_0} = R^{TOA}_{atm+surf} + T_{dir}(\theta_v)\,T_{dir}(\theta_s)\,R^{BOA}_{sunglint}
+ T_g(\theta_v)\,t_{L_u}(\theta_v)\,t_{E_d}(\theta_s)\,\pi R^{BOA}_{rs}
\label{eq:toa}
\end{equation}'''
EQ_UNC = r'''\begin{equation}
\sigma\left(R^{TOA}_{tot}\right) = \sqrt{\left(\frac{\partial R^{TOA}_{tot}}{\partial R_{rs}}\,\sigma(R_{rs})\right)^2
+ \left(\frac{\partial R^{TOA}_{tot}}{\partial aot}\,\sigma(aot)\right)^2}
\label{eq:unc}
\end{equation}'''


EQ_MD5 = {'4058de4c5a81c83e0f97158fa19fd58c': 'toa', 'cb122ff44521f2087c686241745b9fc6': 'unc'}


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# brace-aware helpers
# ------------------------------------------------------------------------------------------------------------------
def match_brace(s, i):
    '''Index after the group starting at s[i] == "{".'''
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == '\\':
            continue
        if c == '{' and (j == 0 or s[j - 1] != '\\'):
            depth += 1
        elif c == '}' and (j == 0 or s[j - 1] != '\\'):
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError('unbalanced braces at ' + s[i:i + 80])


def replace_command(s, name, func, nopt=0):
    '''Replace each \\name[opt]{arg} (nopt optional [..] groups before one mandatory group) by func(opts, arg).'''
    out, pos = [], 0
    pat = re.compile(r'\\' + name + r'(?![A-Za-z])')
    while True:
        m = pat.search(s, pos)
        if not m:
            out.append(s[pos:])
            return ''.join(out)
        j = m.end()
        opts = []
        for _ in range(nopt):
            if j < len(s) and s[j] == '[':
                k = s.index(']', j)
                opts.append(s[j + 1:k])
                j = k + 1
        if j >= len(s) or s[j] != '{':
            out.append(s[pos:m.end()])
            pos = m.end()
            continue
        k = match_brace(s, j)
        out.append(s[pos:m.start()])
        out.append(func(opts, s[j + 1:k - 1]))
        pos = k


def texorpdf(t):
    '''\\texorpdfstring{tex}{pdf} -> tex.'''
    while '\\texorpdfstring{' in t:
        i = t.index('\\texorpdfstring{')
        j = match_brace(t, i + len('\\texorpdfstring'))
        k = match_brace(t, j) if j < len(t) and t[j] == '{' else j
        t = t[:i] + t[i + len('\\texorpdfstring') + 1:j - 1] + t[k:]
    return t


def strip_wrappers(t):
    for cmd in ['textbf', 'emph', 'ul', 'textit']:
        t = replace_command(t, cmd, lambda o, a: a)
    return t.strip()


# ------------------------------------------------------------------------------------------------------------------
# steps
# ------------------------------------------------------------------------------------------------------------------
def hyperrefs(body):
    '''Internal links of the Word template: figure anchors and references, pseudo-headings, plain text.'''
    body = re.sub(r'\\protect\\phantomsection\\label\{[^}]*\}\{\}', '', body)

    def ref(opts, arg):
        key = opts[0] if opts else ''
        if key.startswith('figur_'):                      # caption anchor
            return f'@@CAP:{key[6:]}@@' + arg
        if key.startswith('fig_'):                        # reference to a figure in the text
            return re.sub(r'(\d+)', r'\\ref{fig:' + key[4:] + '}', arg, count=1)
        return f'@@HREF@@{arg}@@/HREF@@'
    body = replace_command(body, 'hyperref', ref, nopt=1)

    # paragraph made of one link only (optionally numbered "1 --", optionally ending with ":"): pseudo-heading
    def head(m):
        text = strip_wrappers(m.group(2)).rstrip(':').strip()
        return f'@@HEAD:{text}@@'
    body = re.sub(r'(?m)(?<=\n\n)(\d+ -- )?@@HREF@@([^@\n&]*?)@@/HREF@@:?[ \t]*(?=\n\n)', head, body)
    return body.replace('@@HREF@@', '').replace('@@/HREF@@', '')


def headings(body, sub):
    '''Sections: clean titles, single-item numbered lists, pseudo-headings.'''
    def clean_title(t):
        t = texorpdf(t)
        t = re.sub(r'^\d+(\.\d+)*\.?\s+', '', t.strip())             # numbers typed in the Word headings
        t = strip_wrappers(t.replace('\\hfill\\break', ' '))
        t = re.sub(r'\s*-{1,2}\s*(\d+(\.\d+)*|Note)\s*$', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def sec(level):
        def f(opts, arg):
            return '\\' + level + '{' + clean_title(arg) + '}'
        return f
    for level in ['section', 'subsection', 'subsubsection']:
        body = replace_command(body, level, sec(level))
    body = re.sub(r'\\label\{[^}]*\}', lambda m: '' if m.group(0).startswith('\\label{') and
                  not m.group(0).startswith(('\\label{fig:', '\\label{tab:', '\\label{eq:')) else m.group(0), body)
    body = re.sub(r'\\(sub)*section\{\}\s*', '', body)               # empty headings

    # numbered list with a single item: a section of the template
    def single(m):
        return '\\section{' + clean_title(m.group(1)) + '}'
    body = re.sub(r'\\begin\{enumerate\}\n\\def\\labelenumi\{[^\n]*\}\n(?:\\setcounter\{enumi\}\{\d+\}\n)?'
                  r'\\item\n\s+([^\n]+)\n\\end\{enumerate\}', single, body)
    body = re.sub(r'@@HEAD:(.*?)@@', lambda m: '\\' + sub + '*{' + m.group(1) + '}', body)
    # bold paragraph of the TN used as heading (e.g. "Document properties")
    return body


def tables(body):
    '''Tables without minipages; empty rows of the Word templates kept.'''
    body = re.sub(r'\\begin\{minipage\}\[b\]\{\\linewidth\}\\(?:raggedright|centering|raggedleft)\n?(.*?)\n?'
                  r'\\end\{minipage\}', lambda m: m.group(1).strip(), body, flags=re.S)
    return body


def graphics_width(opts):
    m = re.search(r'width=([\d.]+)in', opts or '')
    w = min(float(m.group(1)) / TEXTWIDTH_IN, 1.0) if m else 1.0
    return f'width={w:.2f}\\linewidth'


def figures(body, eq_images):
    '''Images and their captions as figures; table captions; equations given as images.'''
    def inc(opts, arg):
        name = os.path.basename(arg)
        if name in eq_images:
            return f'@@EQ:{eq_images[name]}:{arg}@@'
        return f'@@IMG[{graphics_width(opts[0] if opts else "")}]{{{arg}}}@@'
    body = replace_command(body, 'includegraphics', inc, nopt=1)

    # quotes of the Word body (indented paragraphs) unwrapped; the contact block before the first section kept
    head, sep, rest = body.partition('\\section{')
    rest = re.sub(r'\\(begin|end)\{quote\}', '', rest)
    body = head + sep + rest
    paras = re.split(r'\n\s*\n', body)
    out, i, nfig = [], 0, 0
    img_only = re.compile(r'^(\s*@@IMG\[[^\]]*\]\{[^}]*\}@@\s*)+$')
    cap_re = re.compile(r'^\s*(?:\\(?:emph|textbf)\{)*\s*(?:@@CAP:(\w+)@@)?\s*(Figure|Fig|Table)\.?(?:~|\s)?'
                        r'(?:\d+(?:-\d+)?)?\s*~?\s*[.:]\s*(.*?)\}*\s*$', re.S)

    def caption_text(t):
        return strip_wrappers(t).rstrip('}').strip()

    while i < len(paras):
        p = paras[i]
        # equation images
        m = re.match(r'^\s*@@EQ:(\w+):([^@]+)@@\s*$', p)
        if m:
            kind, path = m.groups()
            if kind == 'toa':
                out.append(EQ_TOA + '\n\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.95\\linewidth]{'
                           + path + '}\n\\caption{Terms of the TOA reflectance and their sources.}\n'
                           '\\label{fig:toa_terms}\n\\end{figure}')
            else:
                out.append(EQ_UNC)
            i += 1
            continue
        # image(s) with the caption in the same paragraph
        m = re.match(r'^((?:\s*@@IMG\[[^\]]*\]\{[^}]*\}@@)+)(.+)$', p, re.S)
        if m and cap_re.match(m.group(2)):
            paras[i:i + 1] = [m.group(1), m.group(2)]
            p = paras[i]
        if img_only.match(p):
            imgs = [p]
            j = i + 1
            while j < len(paras) and img_only.match(paras[j]):
                imgs.append(paras[j])
                j += 1
            c = cap_re.match(paras[j]) if j < len(paras) else None
            graphics = re.findall(r'@@IMG\[([^\]]*)\]\{([^}]*)\}@@', '\n'.join(imgs))
            incl = '\n'.join(f'\\includegraphics[{o}]{{{f}}}' for o, f in graphics)
            if c and c.group(2).startswith('Fig'):
                nfig += 1
                label = c.group(1) or f'auto{nfig}'
                out.append('\\begin{figure}[H]\n\\centering\n' + incl.replace('}\n\\inc', '}\\\\[4pt]\n\\inc') +
                           '\n\\caption{' + caption_text(c.group(3)) + '}\n\\label{fig:' + label + '}\n\\end{figure}')
                i = j + 1
            else:
                out.append('\\begin{center}\n' + incl.replace('}\n\\inc', '}\\\\[4pt]\n\\inc') + '\n\\end{center}')
                i = j
            continue
        # table caption before an image or a longtable
        c = cap_re.match(p)
        if c and c.group(2) == 'Table' and i + 1 < len(paras):
            nxt = paras[i + 1]
            text = caption_text(c.group(3))
            if img_only.match(nxt):
                graphics = re.findall(r'@@IMG\[([^\]]*)\]\{([^}]*)\}@@', nxt)
                incl = '\n'.join(f'\\includegraphics[{o}]{{{f}}}' for o, f in graphics)
                out.append('\\begin{table}[H]\n\\centering\n\\caption{' + text + '}\n' + incl + '\n\\end{table}')
                i += 2
                continue
            if nxt.lstrip().startswith('\\begin{longtable}'):
                out.append('\\begin{center}\n\\captionof{table}{' + text + '}\n\\end{center}')
                i += 1
                continue
        out.append(p)
        i += 1
    body = '\n\n'.join(out)
    # leftover markers (images inside other text, captions without image)
    body = re.sub(r'@@IMG\[([^\]]*)\]\{([^}]*)\}@@', r'\\includegraphics[\1]{\2}', body)
    body = re.sub(r'@@CAP:\w+@@', '', body)
    return body


def misc(body):
    body = replace_command(body, 'ul', lambda o, a: a)
    body = replace_command(body, 'href', lambda o, a: a, nopt=0) if False else body
    body = re.sub(r'\\href\{([^}]*)\}\{\\ul\{(.*?)\}\}', r'\\href{\1}{\2}', body)
    body = body.replace('\\textbf{(Please hover mouse over hyperlinks for descriptions)}\n\n', '')
    body = re.sub(r'\\textbf\{RadCalNet country codes \\href\{[^}]*\}\{Link:\}\}\n\n', '', body)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return body


# ------------------------------------------------------------------------------------------------------------------
# front matter, header and footer
# ------------------------------------------------------------------------------------------------------------------
def header_images(docx, media):
    '''Images of the Word headers and footers, copied to media/ (pandoc drops the headers).'''
    out = {}
    with zipfile.ZipFile(docx) as z:
        for part in sorted(n for n in z.namelist() if re.match(r'word/_rels/(header|footer)\d+\.xml\.rels', n)):
            kind = 'header' if 'header' in part else 'footer'
            for target in re.findall(r'Target="(media/[^"]+)"', z.read(part).decode()):
                name = f'{kind}_{os.path.basename(target)}'
                with z.open('word/' + target) as src, open(os.path.join(media, name), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                out.setdefault(kind + os.path.basename(part)[len(kind):], []).append('media/' + name)
    return out


def header_text(docx):
    with zipfile.ZipFile(docx) as z:
        texts = {}
        for n in z.namelist():
            if re.match(r'word/(header|footer)\d+\.xml$', n):
                xml = re.sub(r'<w:tab/>|</w:p>', '<w:t> \u00b7 </w:t>', z.read(n).decode())
                runs = re.findall(r'<w:t(?: [^>]*)?>([^<]*)</w:t>', xml)
                t = re.sub(r'\s+', ' ', ''.join(runs))
                t = re.sub(r'(\s*\u00b7\s*)+', ' \u00b7 ', t).strip(' \u00b7')
                texts[os.path.basename(n)] = t.replace('\u00b7', '--')
    return texts


def front_qu(body, kind, logos, texts, docx):
    '''Title page of the RadCalNet questionnaire (Q) and uncertainty statement (U).'''
    head, sep, rest = body.partition('\\section{')
    t = re.sub(r'\\begin\{center\}.*?\\end\{center\}', '', head, flags=re.S)  # logo of the first page
    def field(pat):
        m = re.search(pat, t, re.S)
        return strip_wrappers(m.group(1)).strip() if m else ''
    ref = re.sub(r'\s+', '', field(r'CEOS Reference:\s*(.*?)\n\n').replace('\\textbf{', '').replace('}', ''))
    site = field(r'Name of site:\}?\s*(?:\\textbf\{)?(.*?)\}?\n\n')
    contact = re.search(r'\\begin\{quote\}(.*?)\\end\{quote\}', t, re.S)
    contact = contact.group(1).strip() if contact else ''
    contact = '\\\\\n'.join(x.strip() for x in contact.split('\n\n') if x.strip())
    title = field(r'\\textbf\{((?:Uncertainty Analysis Statement|Site description)[^\n]*?)\}\n')
    author = field(r'Author\(s\):\s*(.*?)\}?\s*(?:\\emph\{)?Date:')
    date = field(r'Date:\s*(.*?)\}')
    doctype = 'RadCalNet site uncertainty statement' if kind == 'U' else 'RadCalNet site description'
    hdr = next((v for k, v in texts.items() if k.startswith('header')), '')
    ftr = next((v for k, v in texts.items() if k.startswith('footer')), '')
    first_logo = re.search(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', head)
    pieces = ['\\begin{titlepage}', '\\centering']
    if first_logo:
        pieces.append(f'\\includegraphics[width=0.55\\linewidth]{{{first_logo.group(1)}}}\\par\\vspace{{1.5cm}}')
    pieces += [f'{{\\LARGE\\bfseries {doctype}\\par}}\\vspace{{0.6cm}}',
               f'{{\\large CEOS reference: {ref}\\par}}\\vspace{{1.2cm}}' if ref else '',
               f'{{\\Large {title}\\par}}\\vspace{{0.8cm}}' if title else '',
               f'{{\\large Site: {site}\\par}}\\vspace{{0.6cm}}' if site else '',
               f'{{\\large {author}\\par}}' if author else '',
               f'{{\\large {date}\\par}}\\vspace{{1.5cm}}' if date else '',
               '\\begin{minipage}{0.8\\linewidth}\\raggedright\n\\textbf{Point of contact}\\\\[4pt]\n' + contact +
               '\n\\end{minipage}' if contact else '',
               '\\vfill', '\\end{titlepage}']
    front = '\n'.join(x for x in pieces if x)
    # the Word table of contents of the questionnaires
    rest = sep + rest
    rest = re.sub(r'\\section\{Table of Contents\}.*?(?=\\section\{)',
                  r'\\tableofcontents\n\\newpage\n\n', rest, count=1, flags=re.S)
    if kind == 'U':
        rest = re.sub(r'^\\section\{Purpose of Note\}', r'\\section*{Purpose of Note}', rest)
    hdr_left = logos.get('header1.xml.rels', [None])[0]
    hdr_right = logos.get('header1.xml.rels', [None, None])[-1]
    fancy = ['\\pagestyle{fancy}', '\\fancyhf{}',
             f'\\fancyhead[L]{{\\includegraphics[height=0.9cm]{{{hdr_left}}}}}' if hdr_left else '',
             f'\\fancyhead[R]{{\\includegraphics[height=0.9cm]{{{hdr_right}}}}}' if hdr_right and hdr_right != hdr_left else '',
             f'\\fancyhead[C]{{\\parbox[b]{{0.5\\textwidth}}{{\\centering\\scriptsize\\bfseries {latex_escape(hdr)}}}}}'
             if hdr else '',
             f'\\fancyfoot[L]{{\\footnotesize {latex_escape(ftr)}}}' if ftr else '',
             '\\fancyfoot[R]{\\footnotesize\\thepage}',
             '\\setlength{\\headheight}{1.4cm}']
    return front, rest, '\n'.join(x for x in fancy if x)


def front_tn(body, logos, texts):
    '''Title page of the technical note: title, then the tables of the Word cover pages.'''
    m = re.match(r'\s*\\begin\{longtable\}.*?\\toprule\\noalign\{\}\n(.*?)\\midrule.*?\\end\{longtable\}', body, re.S)
    title_lines = [strip_wrappers(x) for x in m.group(1).split('\n') if x.strip()] if m else []
    rest = body[m.end():] if m else body
    logo = logos.get('header1.xml.rels', [None])[0]
    pieces = ['\\begin{titlepage}', '\\centering']
    if logo:
        pieces.append(f'\\includegraphics[width=0.35\\linewidth]{{{logo}}}\\par\\vspace{{2cm}}')
    if title_lines:
        pieces.append(f'{{\\LARGE\\bfseries {title_lines[0]}\\par}}\\vspace{{0.8cm}}')
        pieces += [f'{{\\Large {x}\\par}}\\vspace{{0.5cm}}' for x in title_lines[1:]]
    pieces += ['\\vfill', '{\\large ACTION-TN-101-MAG, edition 2.0, 18/08/2025\\par}', '\\end{titlepage}']
    # bold paragraphs used as table titles on the cover pages
    rest = re.sub(r'(?m)^\\textbf\{(Document properties|Distribution List|Document Change Record)\}$',
                  r'\\subsection*{\1}', rest)
    rest = re.sub(r'\\textbf\{Table of contents\}.*?(?=\\section\{Introduction\})',
                  r'\\tableofcontents\n\\newpage\n\n', rest, count=1, flags=re.S)
    hdr = re.sub(r'DOCPROPERTY "\w+"', '', texts.get('header2.xml', '')).strip()
    hdr = re.sub(r'\s+\S+$', '', hdr) if len(hdr) > 120 else hdr
    hdr_logo = logos.get('header2.xml.rels', [logo])[0]
    fancy = ['\\pagestyle{fancy}', '\\fancyhf{}',
             f'\\fancyhead[L]{{\\includegraphics[height=0.8cm]{{{hdr_logo}}}}}' if hdr_logo else '',
             '\\fancyhead[R]{\\footnotesize Demonstrating the Feasibility of a Potential Extension\\\\ '
             'of RadCalNet to Oceanic Sites}',
             '\\fancyfoot[L]{\\footnotesize R\\\'ef.: ACTION-TN-101-MAG}',
             '\\fancyfoot[C]{\\footnotesize 18/08/2025}',
             '\\fancyfoot[R]{\\footnotesize\\thepage}',
             '\\setlength{\\headheight}{1.4cm}']
    return '\n'.join(pieces), rest, '\n'.join(x for x in fancy if x)


def latex_escape(t):
    t = re.sub(r'-\s+', '-', re.sub(r'\s+-(?=[A-Z])', '-', t))     # "WGCV- RadCalNet" of the Word headers
    return t.replace('&quot;', '"').replace('_', '\\_').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')


PREAMBLE_EXTRA = r'''\setcounter{secnumdepth}{3}
\usepackage{fancyhdr}
\usepackage{amsmath}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{newunicodechar}
\newunicodechar{⁰}{\textdegree}
\setlength{\emergencystretch}{3em}
\usepackage[margin=2.2cm,headsep=0.6cm]{geometry}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color[HTML]{1F4E79}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\large\bfseries\color[HTML]{1F4E79}}{\thesubsection}{0.8em}{}
'''


# corrections of the Word originals (text copied from the Venise documents into the Galata ones)
CORRECTIONS = {
    'QA4EO-WGCV-RadCalNet-Galata-U-v1': [
        ('Uncertainty Analysis Statement -- AERONET-OC - Venise', 'Uncertainty Analysis Statement -- AERONET-OC - Galata'),
        ('provided from AERONET-OC Venise site', 'provided from AERONET-OC Galata site'),
        ('processing chain for AERONET-OC Venise site', 'processing chain for AERONET-OC Galata site'),
        ('TOA reflectance for AERONET-OC Venise site', 'TOA reflectance for AERONET-OC Galata site'),
    ],
}


def main(name):
    folder = os.path.join(HERE, name)
    docx = os.path.join(HERE, '..', name + '.docx')
    src = os.path.join(folder, 'pandoc.tex')
    if not os.path.exists(src):
        shutil.copy(os.path.join(folder, 'main.tex'), src)
    tex = open(src).read()
    pre, _, body = tex.partition('\\begin{document}')
    body = body.replace('\\end{document}', '')
    # preamble of pandoc: no title, no geometry (set below)
    pre = re.sub(r'(?m)^\\(title|author|date)\{.*?\}\s*$', '', pre)
    pre = re.sub(r'\\title\{.*?\}\}\n', '', pre, flags=re.S)
    pre = re.sub(r'(?m)^\\usepackage\[[^\]]*\]\{geometry\}\n', '', pre)
    pre = re.sub(r'(?m)^\\usepackage\{(float|caption|newunicodechar)\}\n|^\\floatplacement\{figure\}\{H\}\n|'
                 r'^\\newunicodechar\{.\}\{[^\n]*\}\n|^\\setlength\{\\emergencystretch\}\{3em\}\n', '', pre)
    body = body.replace('\\maketitle', '')

    kind = 'TN' if name.startswith('ACTION') else name.split('-')[-2]   # Q or U
    media = os.path.join(folder, 'media')
    logos = header_images(docx, media)
    texts = header_text(docx)

    # equations given as images (same images in the Venise and Galata statements)
    eq_images = {f: EQ_MD5[md5(os.path.join(media, f))] for f in os.listdir(media)
                 if md5(os.path.join(media, f)) in EQ_MD5}

    body = hyperrefs(body)
    body = headings(body, 'subsection' if kind == 'U' else 'subsubsection')
    body = tables(body)
    body = figures(body, eq_images)
    body = misc(body)
    if kind == 'TN':
        front, body, fancy = front_tn(body, logos, texts)
    else:
        front, body, fancy = front_qu(body, kind, logos, texts, docx)
    numbering = ''
    if kind in ('TN', 'Q'):      # tables and figures numbered per section as in the Word documents ("Table 4-1")
        numbering = ('\\counterwithin{figure}{section}\\counterwithin{table}{section}\n'
                     '\\renewcommand{\\thefigure}{\\thesection-\\arabic{figure}}\n'
                     '\\renewcommand{\\thetable}{\\thesection-\\arabic{table}}\n')
    tex = pre.rstrip() + '\n' + PREAMBLE_EXTRA + numbering + fancy + '\n\n\\begin{document}\n\n' + front + '\n\n' + \
        body.strip() + '\n\n\\end{document}\n'
    for old, new in CORRECTIONS.get(name, []):
        if old not in tex:
            print(f'{name}: correction not applied, text not found: {old!r}')
        tex = tex.replace(old, new)
    open(os.path.join(folder, 'main.tex'), 'w').write(tex)
    print(f'{name}: equations {eq_images}, header images {logos}')


if __name__ == '__main__':
    main(sys.argv[1].rstrip('/'))
