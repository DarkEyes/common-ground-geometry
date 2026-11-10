"""
config.py -- paths used by every script in this package.

Every path is relative to the folder that contains this file, so the scripts
work from any working directory and on any machine. Two environment variables
can override the data locations:

    CG_WCS_DIR   folder with the WCS raw files (default: data/wcs)
    CG_XKCD_DB   SQLite file built from the xkcd survey (default: data/xkcd/main.db)

Importing this module also changes the working directory to the package
folder, so that relative paths such as results/ always resolve the same way.
"""
import os
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
os.chdir(PKG)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

DATA = os.path.join(PKG, "data")
WCS_DIR = os.environ.get("CG_WCS_DIR", os.path.join(DATA, "wcs"))
XKCD_DIR = os.path.join(DATA, "xkcd")
XKCD_DB = os.environ.get("CG_XKCD_DB", os.path.join(XKCD_DIR, "main.db"))
RESULTS = os.path.join(PKG, "results")
FIGURES = os.path.join(PKG, "paper_figures")
