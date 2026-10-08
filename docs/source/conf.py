# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the package importable without installation (local builds);
# on Read the Docs the package is also pip-installed (see .readthedocs.yaml).
DOCS_SOURCE = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import radcalnet_oc

# -- Project information -----------------------------------------------------

project = 'radcalnet_oc'
copyright = '2026, Tristan Harmel'
author = 'Tristan Harmel'
version = radcalnet_oc.__version__
release = radcalnet_oc.__version__
today_fmt = '%Y-%m-%d'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
    'myst_nb',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = 'both'          # the parameters are documented in __init__
autodoc_typehints = 'none'          # types are given in the docstrings
autodoc_member_order = 'bysource'
# 'members' is set in the autosummary templates (_templates/autosummary/) to
# avoid documenting objects twice
add_module_names = False

# The docstrings use reST fields (":param x:"); napoleon also converts the
# NumPy sections ("Parameters", "Notes") if any.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# labelled equations are numbered by page and cited with :eq:
math_eqref_format = '({number})'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

html_theme_options = {
    'repository_url': 'https://github.com/Tristanovsk/radcalnet_oc',
    'repository_branch': 'main',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
    'navigation_with_keys': True,
    'show_toc_level': 2,
    'logo': {
        'image_light': '_static/radcalnet-oc-logo.png',
        'image_dark': '_static/radcalnet-oc-logo-dark.png',
    },
}

html_title = ''
html_logo = '_static/radcalnet-oc-logo.png'
html_favicon = '_static/radcalnet-oc-favicon.png'

html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sourcelink = False
html_last_updated_fmt = today_fmt

htmlhelp_basename = 'radcalnet_oc_doc'

# -- MyST / notebook rendering -----------------------------------------------

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'smartquotes',
]
myst_heading_anchors = 3

# The tutorial notebooks need the look-up tables that are not available on
# Read the Docs: they are rendered with the outputs stored in the notebooks,
# not executed.
nb_execution_mode = 'off'
nb_merge_streams = True
suppress_warnings = ['mystnb.unknown_mime_type', 'myst.header']
