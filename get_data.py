"""
get_data.py -- download the two datasets and check them against the
checksums in results/input_checksums.txt.

    python get_data.py          # both datasets
    python get_data.py --wcs    # only the World Color Survey (enough for the figures)

World Color Survey (WCS): the five raw files from a byte-identical mirror of
the WCS Data Archives (Cook, Kay, Regier), https://github.com/jvosten/wcs.
xkcd color survey: colorsurvey.tar.gz from https://xkcd.com/color/, whose
mainsurvey_sqldump.txt is loaded into SQLite.
Files that are already present are not downloaded again.
"""
import hashlib
import os
import sqlite3
import sys
import tarfile
import urllib.request

import config

WCS_FILES = ["term.txt", "foci-exp.txt", "cnum-vhcm-lab-new.txt", "dict.txt", "spkr-lsas.txt"]
WCS_URL = "https://raw.githubusercontent.com/jvosten/wcs/main/data-raw/"
XKCD_URL = "https://xkcd.com/color/colorsurvey.tar.gz"


def expected_checksums():
    out = {}
    path = os.path.join(config.RESULTS, "input_checksums.txt")
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            out[os.path.basename(parts[1]).replace("colorsurvey_tar.gz", "colorsurvey.tar.gz")] = parts[0]
    return out


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def check(path, expected):
    name = os.path.basename(path)
    got = sha16(path)
    if name in expected and expected[name] != got:
        print("WARNING: %s has checksum %s, expected %s; results may differ" % (name, got, expected[name]))
    else:
        print("ok      %s" % name)


def download(url, path):
    print("download %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def get_wcs(expected):
    os.makedirs(config.WCS_DIR, exist_ok=True)
    for name in WCS_FILES:
        path = os.path.join(config.WCS_DIR, name)
        if not os.path.exists(path):
            download(WCS_URL + name, path)
        check(path, expected)


def get_xkcd(expected):
    os.makedirs(config.XKCD_DIR, exist_ok=True)
    if os.path.exists(config.XKCD_DB):
        print("ok      %s (already built)" % config.XKCD_DB)
        return
    tgz = os.path.join(config.XKCD_DIR, "colorsurvey.tar.gz")
    if not os.path.exists(tgz):
        download(XKCD_URL, tgz)
    check(tgz, expected)
    dump = os.path.join(config.XKCD_DIR, "mainsurvey_sqldump.txt")
    if not os.path.exists(dump):
        with tarfile.open(tgz) as t:
            t.extract("mainsurvey_sqldump.txt", config.XKCD_DIR)
    print("build   %s (a few minutes)" % config.XKCD_DB)
    con = sqlite3.connect(config.XKCD_DB + ".part")
    with open(dump, encoding="utf-8", errors="replace") as f:
        con.executescript(f.read())
    con.commit()
    con.close()
    os.replace(config.XKCD_DB + ".part", config.XKCD_DB)
    os.remove(dump)


if __name__ == "__main__":
    exp = expected_checksums()
    get_wcs(exp)
    if "--wcs" not in sys.argv:
        get_xkcd(exp)
