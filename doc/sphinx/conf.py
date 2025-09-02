# Minimal Sphinx configuration for nerva_jax docs
import os
import sys
from datetime import datetime

# Add src to path so autodoc can import the package
sys.path.insert(0, os.path.abspath('../../src'))

project = 'nerva_jax'
author = 'Wieger Wesselink and contributors'
current_year = datetime.now().year
copyright = f'{current_year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

# Avoid heavy imports during autodoc builds on CI
autodoc_mock_imports = [
    'jax',
    'jax.numpy',
    'sklearn',
]

# Generate autosummary pages automatically
autosummary_generate = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = 'sphinx_rtd_theme'

# Keep output simple and fast to build
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
