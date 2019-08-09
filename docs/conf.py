import sys, os
sys.path.insert(0, os.path.abspath('.'))


extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    ]


# basic options
html_theme = 'bootstrap'
html_title = "MTUQ Documentation"
master_doc = 'index'
source_suffix = '.rst'
templates_path = ['_templates']


# theme options
html_theme_options = {
    'navbar_title': 'MTUQ',
    'navbar_site_name': 'Documentation',
    'source_link_position': 'none',
    'bootswatch_theme': '',
    'bootstrap_version': '3',
}

# extension options
autosummary_generate = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

