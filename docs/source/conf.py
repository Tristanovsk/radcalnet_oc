# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'radcalnet_oc'
copyright = '2025, Tristan Harmel'
author = 'Tristan Harmel'
release = '0.0.1'
today_fmt = "%Y-%m-%d"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'myst_nb',
    'IPython.sphinxext.ipython_console_highlighting']

## Include Python objects as they appear in source files
## Default: alphabetically ('alphabetical')
autodoc_member_order = 'bysource'
## Default flags used by autodoc directives
autodoc_default_flags = ['members', 'show-inheritance']
## Generate autodoc stubs with summaries from code
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output -------------------------------------------------
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/Tristanovsk/radcalnet_oc",
    "repository_branch": "master",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "navigation_with_keys":True,
    "path_to_docs": "docs",
}


html_logo = "_static/radcalnet-oc-logo.png"
html_title = ""

html_favicon = "_static/radcalnet-oc-favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.

html_show_sourcelink = False

html_last_updated_fmt = today_fmt

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "radcalnet_oc_doc"

# -------------------------------
# For Jupyter notebook rendering
# --------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
]

# Autodoc
autodoc_default_options = {
    'member-order': 'groupwise',
    'show-inheritance': True,
}

# Notebook integration parameters
nbsphinx_execute = 'auto'
#nb_execution_mode = "off"
nb_execution_mode = "cache"
nb_execution_timeout = -1
nb_execution_allow_errors = True

# Manage new READTHEDOCS output mechanism
cache_path = os.getenv('READTHEDOCS_OUTPUT')
if cache_path is not None:
    nb_execution_cache_path = f"{cache_path}/../docs/build/.jupyter_cache"

# Merge stderr and stdout
nb_merge_streams = True

