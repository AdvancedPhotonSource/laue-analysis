"""Sphinx configuration for the laueanalysis documentation."""

from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    metadata = tomllib.load(pyproject_file)["project"]

project = metadata["name"]
author = "Advanced Photon Source"
copyright = "2026, Advanced Photon Source"
release = metadata["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "Path"),
    ("py:class", "np.ndarray"),
    ("py:class", "laueanalysis.indexing._liblaue.DetectorGeometry"),
]

autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "Path": "pathlib.Path",
    "np.ndarray": "numpy.ndarray",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "laueanalysis documentation"
html_logo = "_static/laueanalysis-logo.svg"
html_favicon = "_static/laueanalysis-logo.svg"
html_baseurl = "https://advancedphotonsource.github.io/laue-analysis/"
html_theme_options = {
    "github_url": "https://github.com/AdvancedPhotonSource/laue-analysis",
    "navbar_align": "left",
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "AdvancedPhotonSource",
    "github_repo": "laue-analysis",
    "github_version": "main",
    "doc_path": "docs",
}
