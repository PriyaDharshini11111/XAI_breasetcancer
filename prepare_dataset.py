#!/usr/bin/env python3
"""Convert Mini-DDSM / CBIS-DDSM (or any dataset) into the folder layout used by mainCode.ipynb.

    <out>/<class>/<name>.png            image (converted to 8-bit PNG)
    <out>/<class>/<name>_mask.png       binary lesion mask (0 / 255); omitted for classes without lesions

Input is a CSV manifest that YOU build from the dataset's own metadata files (the raw folder structures differ between
the TCIA, Kaggle and project-page copies, so no layout is guessed here). Columns:

    image_path   path of the image (png / jpg / tif; DICOM if pydicom is installed)
    class        class folder name, lower-case: benign | malignant | normal | benign_without_callback
    mask_path    (optional) mask or outline image; empty for normal images
    name         (optional) output file stem; defaults to the image file stem

--mask-mode
    binary   mask_path already holds a lesion mask (CBIS-DDSM ROI masks)           -> thresholded at 127
    outline  mask_path holds an outline curve (Mini-DDSM)                           -> the outline is closed and filled
    none     no masks (classification only)

Example:
    python prepare_dataset.py --manifest minidd_manifest.csv --out data/MiniDDSM --mask-mode outline
"""
import argparse, os, sys
import cv2
import numpy as np
import pandas as pd


def read_gray(path):
    if path.lower().endswith((".dcm", ".dicom")):
        import pydicom
        a = pydicom.dcmread(path).pixel_array.astype(np.float32)
        a = (255 * (a - a.min()) / max(a.max() - a.min(), 1e-6)).astype(np.uint8)
        return a
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def binary_mask(path, shape):
    m = read_gray(path)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((m > 127) * 255).astype(np.uint8)


def outline_to_mask(path, shape):
    """Close an outline curve and fill its interior (union of all closed contours)."""
    o = read_gray(path)
    if o.shape != shape:
        o = cv2.resize(o, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    o = ((o > 127) * 255).astype(np.uint8)
    o = cv2.morphologyEx(o, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # bridge small gaps
    cnts, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, cnts, -1, 255, thickness=cv2.FILLED)
    return m


def polygon_to_mask(coords, shape):
    """coords: iterable of (x, y) path points -> filled binary mask (for datasets that ship ROI path coordinates)."""
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(coords, np.int32).reshape(-1, 1, 2)], 255)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-mode", choices=["binary", "outline", "none"], default="binary")
    a = ap.parse_args()
    mf = pd.read_csv(a.manifest)
    for col in ("image_path", "class"):
        if col not in mf:
            sys.exit(f"manifest needs a '{col}' column")
    n = 0
    for row in mf.to_dict("records"):
        cls = str(row["class"]).strip().lower()
        nm = row.get("name")
        name = nm if isinstance(nm, str) and nm.strip() else os.path.splitext(os.path.basename(row["image_path"]))[0]
        d = os.path.join(a.out, cls); os.makedirs(d, exist_ok=True)
        img = read_gray(row["image_path"])
        cv2.imwrite(os.path.join(d, f"{name}.png"), img)
        mp = row.get("mask_path")
        if a.mask_mode != "none" and isinstance(mp, str) and mp.strip():
            fn = binary_mask if a.mask_mode == "binary" else outline_to_mask
            cv2.imwrite(os.path.join(d, f"{name}_mask.png"), fn(mp, img.shape))
        n += 1
    print(f"converted {n} images into {a.out}")


if __name__ == "__main__":
    main()
