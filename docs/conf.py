#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Make the project importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

todo_include_todos = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

source_suffix = ".rst"
master_doc = "index"

project = "XFusion"
author = "Argonne National Laboratory"
copyright = f"2024, {author}"

# Version info
version = open(os.path.join("..", "VERSION")).read().strip()
release = version

pygments_style = "sphinx"
show_authors = False

# -- HTML output ----------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_nav_header_background": "#4f8fb8ff",
    "collapse_navigation": False,
    "logo_only": True,
}

htmlhelp_basename = project + "doc"

# -- Autodoc: mock only heavy/optional deps and compiled ops ---------------

# Do NOT mock stdlib modules (os, sys, pathlib, etc.) or your own package.
# Mock only things that may be unavailable on RTD / fresh doc envs.
autodoc_mock_imports = [
    "cv2",
    "PIL",
    "skimage",
    "torch",
    "torchvision",

    # inference compiled/custom op chain
    "xfusion.inference.ops.dcn",
    "xfusion.inference.ops.dcn.deform_conv",

    # training compiled/custom op chain (this is what your traceback hits)
    "xfusion.train.basicsr.ops.dcn",
    "xfusion.train.basicsr.ops.dcn.deform_conv",
    "timm",
    "einops",
    "tqdm",
    "natsort",
    "scipy",
    "bgr2ycbcr",
]

# -- LaTeX / man / texinfo (kept minimal) ---------------------------------

latex_documents = [
    ("index", f"{project}.tex", f"{project} Documentation", author, "manual"),
]

man_pages = [
    ("index", project, f"{project} Documentation", [author], 1),
]

texinfo_documents = [
    (
        "index",
        project,
        f"{project} Documentation",
        author,
        project,
        "Data Management: APS Data Management in Python.",
        "Miscellaneous",
    ),
]
