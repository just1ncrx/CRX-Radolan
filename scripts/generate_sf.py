#!/usr/bin/env python3
"""RW-Composite (DWD RADOLAN RW, ODIM-HDF5) -> NATIVES DWD-Raster (stereographisch).

RW = 1-Stunden-Niederschlagssumme in mm.
Kein Warp nach Web Mercator, kein WebP. Ausgabe: rohe uint8-Bytes, keine Dateiendung.
Layout: ysize Zeilen x xsize Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize Bytes (900 x 900 = 810.000).

Jeder Pixel ist ein CODE (0-255), der Client faerbt ihn per Tabelle ein:
    0        unter 0.1 mm/h (kein Regen)          -> transparent
    1 .. 29  Klassen der COLOR_TABLE: Code = 1 + Nummer der Klasse (0 = 0.1-0.5 mm, ...)
    255      keine Daten / ausserhalb              -> transparent

Farbtabelle fuer den Client (radar-colors.js) ausgeben:   python3 rw_native.py --colors

Aufruf:
    python3 rw_native.py                 # neueste RW-Datei in SRC_DIR
    python3 rw_native.py <rw-datei>      # bestimmte Datei
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/sf")
OUT_DIR = Path("output/sf")

# raa01-rw_10000-YYMMDDHHMM-dwd---bin.hdf5  (Zeit = Ende der Stunde)
FILENAME_RE = re.compile(r"raa01-sf_10000-(\d{10})-dwd---bin\.hdf5$")

# Fallbacks, falls Attribute (gain/nodata) in der Datei fehlen (RW: 0.1 mm je Einheit)
RW_DEFAULT_GAIN = 0.1
RW_DEFAULT_NODATA = 65535

MAX_VALID_MM = 500.0        # alles darueber = ungueltig (Stundensumme)
MIN_VISIBLE_MM = 0.1        # darunter: Code 0 (transparent)

NODATA_CODE = 255
STEP_CODE_OFFSET = 1        # Code der ersten Klasse

# Nur Pixel innerhalb der Radar-Reichweite behalten (ausserhalb -> Code 255)
MASK_OUTSIDE_RADAR = True
RADAR_RANGE_KM = 150.0
RADAR_MARGIN_KM = 0.0

RADARS = {
    "asb": dict(name="ASR Borkum",     lat=53.564011, lon=6.748292),
    "boo": dict(name="Boostedt",       lat=54.00438,  lon=10.04687),
    "drs": dict(name="Dresden",        lat=51.12465,  lon=13.76865),
    "eis": dict(name="Eisberg",        lat=49.54066,  lon=12.40278),
    "emd": dict(name="Emden",          lat=53.33872,  lon=7.02377),
    "ess": dict(name="Essen",          lat=51.40563,  lon=6.96712),
    "fbg": dict(name="Feldberg",       lat=47.87361,  lon=8.00361),
    "fld": dict(name="Flechtdorf",     lat=51.31120,  lon=8.802),
    "hnr": dict(name="Hannover",       lat=52.46008,  lon=9.69452),
    "neu": dict(name="Neuhaus",        lat=50.50012,  lon=11.13504),
    "nhb": dict(name="Neuheilenbach",  lat=50.10965,  lon=6.54853),
    "oft": dict(name="Offenthal",      lat=49.9847,   lon=8.71293),
    "pro": dict(name="Prötzel",        lat=52.64867,  lon=13.85821),
    "mem": dict(name="Memmingen",      lat=48.04214,  lon=10.21924),
    "ros": dict(name="Rostock",        lat=54.17566,  lon=12.05808),
    "isn": dict(name="Isen",           lat=48.17470,  lon=12.10177),
    "tur": dict(name="Türkheim",       lat=48.58528,  lon=9.78278),
    "umd": dict(name="Ummendorf",      lat=52.16009,  lon=11.17609),
}

# Klassengrenzen (Legende: 30 Grenzen = 29 Farben), mm je Stunde
LEVELS = [
    0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    60, 70, 80, 90, 100, 110, 130, 150, 250, 350,
]

# Farben aus der Legende (Screenshot), eine je Klasse [LEVELS[i], LEVELS[i+1])
COLORS = [
    "#00C9FF", "#0056FF", "#0000EF", "#BEFFBE", "#98FF98", "#69FF69",
    "#30FF30", "#0AFF0A", "#00DB00", "#00BF00", "#008D00", "#FFFF00",
    "#F0D800", "#EABA00", "#F99C00", "#FF4100", "#FF2700", "#DD0000",
    "#B00000", "#FBC3FB", "#EBABEB", "#DD96DD", "#C674C6", "#BA61BA",
    "#A442A4", "#861586", "#F8F8F8", "#AFAFAF", "#6F6F6F",
]
assert len(LEVELS) == len(COLORS) + 1
assert STEP_CODE_OFFSET + len(COLORS) <= NODATA_CODE, "Zu viele Stufen fuer uint8"

BERLIN = ZoneInfo("Europe/Berlin")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def parse_filename(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Dateiname passt nicht zu 'raa01-sf_10000-YYMMDDHHMM-dwd---bin.hdf5': {filename}")
    return datetime.strptime(m.group(1), "%y%m%d%H%M").replace(tzinfo=timezone.utc)


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_data_dataset(h5file: h5py.File) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der RW-Datei gefunden.")
    return candidates[0]


def read_mm(ds: h5py.Dataset) -> np.ndarray:
    """Dataset -> mm je Stunde (NaN = kein Wert)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", RW_DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", 0.0)))
    nodata = _scalar(attrs.get("nodata", RW_DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None
    quantity = _attr_str(_scalar(attrs.get("quantity", "?")))
    units = _attr_str(_scalar(attrs.get("units", attrs.get("unit", "?"))))

    raw = ds[()]
    values = raw.astype(np.float64) * gain + offset      # RW ist bereits mm je Stunde

    invalid = raw == nodata
    if np.issubdtype(raw.dtype, np.integer):
        invalid |= raw == np.iinfo(raw.dtype).max
    invalid |= ~np.isfinite(values)
    invalid |= values < 0
    invalid |= values > MAX_VALID_MM

    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0
        invalid &= raw != undetect
    values[invalid] = np.nan

    print(f"RW-Dataset: quantity='{quantity}', units='{units}', gain={gain}, offset={offset}, "
          f"nodata={nodata}, undetect={undetect}")
    valid = values[np.isfinite(values)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.2f} .. {valid.max():.2f} mm/h")
    return values


def find_where_group(h5file: h5py.File):
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": _attr_str(where.attrs["projdef"]),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


def radar_coverage_mask(grid: dict, to_proj: Transformer) -> np.ndarray:
    """True = Pixelmitte liegt innerhalb der Reichweite mindestens eines Radars."""
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    xs = ll_x + (np.arange(grid["xsize"]) + 0.5) * grid["xscale"]
    ys = ll_y + (grid["ysize"] - 1 - np.arange(grid["ysize"]) + 0.5) * grid["yscale"]
    xx, yy = np.meshgrid(xs, ys)

    limit = (RADAR_RANGE_KM + RADAR_MARGIN_KM) * 1000.0
    covered = np.zeros(xx.shape, dtype=bool)
    for info in RADARS.values():
        rx, ry = to_proj.transform(info["lon"], info["lat"])
        covered |= np.hypot(xx - rx, yy - ry) <= limit
    return covered


# --------------------------------------------------------------------------- #
# mm -> Code
# --------------------------------------------------------------------------- #
def encode_codes(mm: np.ndarray) -> np.ndarray:
    """mm/h -> uint8-Codes (Klasse i -> STEP_CODE_OFFSET + i)."""
    lower = np.array(LEVELS[:-1], dtype=np.float64)       # untere Grenze je Klasse

    codes = np.zeros(mm.shape, dtype=np.uint8)            # 0 = kein Regen
    finite = np.isfinite(mm)
    codes[~finite] = NODATA_CODE

    visible = finite & (mm >= MIN_VISIBLE_MM - 1e-6)
    idx = np.searchsorted(lower - 1e-6, mm[visible], side="right") - 1
    idx = np.clip(idx, 0, len(lower) - 1)                 # >= 350 bleibt in der letzten Klasse
    codes[visible] = (STEP_CODE_OFFSET + idx).astype(np.uint8)
    return codes


# --------------------------------------------------------------------------- #
# Farbtabelle fuer den Client (radar-colors.js)
# --------------------------------------------------------------------------- #
def print_client_colors() -> None:
    print("  sf: {")
    print("    exact: true,            // jeder Code gilt nur fuer sich")
    print(f"    nodata: [0, {NODATA_CODE}],")
    print("    steps: [")
    for i, c in enumerate(COLORS):
        print(f"      [{STEP_CODE_OFFSET + i}, '{c.lower()}'],   // {LEVELS[i]} - {LEVELS[i + 1]} mm")
    print("    ],")
    print("  },")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if "--colors" in sys.argv:
        print_client_colors()
        return

    if len(sys.argv) > 1:
        src_path = Path(sys.argv[1])
    else:
        candidates = sorted(p for p in SRC_DIR.glob("raa01-sf_10000-*-dwd---bin.hdf5")
                            if FILENAME_RE.match(p.name))
        if not candidates:
            sys.exit(f"Keine RW-Datei in {SRC_DIR} gefunden.")
        src_path = candidates[-1]

    ts = parse_filename(src_path.name)
    print(f"Quelle: {src_path.name}  ({ts:%Y-%m-%d %H:%M} UTC)")

    with h5py.File(src_path, "r") as f:
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HDF5-Datei gefunden.")
        grid = extract_grid_info(where)
        mm = read_mm(find_data_dataset(f))

    h, w = mm.shape
    if (w, h) != (grid["xsize"], grid["ysize"]):
        sys.exit(f"Rastergroesse passt nicht: Array {w}x{h}, where {grid['xsize']}x{grid['ysize']}")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)

    if MASK_OUTSIDE_RADAR:
        covered = radar_coverage_mask(grid, to_proj)
        mm = np.where(covered, mm, np.nan)               # ausserhalb -> NaN -> Code 255
        print(f"Radar-Maske: {covered.mean() * 100:.1f} % des Rasters abgedeckt "
              f"({len(RADARS)} Radare, {RADAR_RANGE_KM:g} km)")

    codes = encode_codes(mm)

    # Speichern: rohe Bytes, keine Dateiendung, Zeile 0 = Norden
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sf_{ts.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {codes.nbytes} Bytes)")

    # Metadaten fuer den Client (einmalig kopieren, aendern sich nicht)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "format": "raw-uint8",
        "width": w,
        "height": h,
        "row_order": "north-to-south",
        "projdef": grid["projdef"],
        "pixel_size_m": grid["xscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, ll_x + w * grid["xscale"], ll_y + h * grid["yscale"]],
    }
    print("Client-Metadaten:", json.dumps(meta))


if __name__ == "__main__":
    main()