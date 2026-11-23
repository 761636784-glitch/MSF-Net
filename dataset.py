"""ENVI BIL -> fixed case manifest -> 100px patches -> pseudo-RGB + 5x5 regional mean/std spectra.

Place reflectance-calibrated ENVI HDR/RAW files in ehcc/ and hgdn/. Run --scan, then --prepare.
Original calibrated RAW cubes are read in read-only memmap mode and never changed.
Only the verified canonical patient manifest is reused; no train/test splitting here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG


LABELS = {"hgdn": 0, "ehcc": 1}  # positive class: eHCC


def read_envi_header(path: Path) -> dict:
    """Parse basic ENVI scalar fields and wavelength array; no extra package needed."""
    raw = path.read_text(encoding="latin-1")
    fields = {}
    for key in ("samples", "lines", "bands", "header offset", "data type", "interleave", "byte order"):
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", raw)
        if match:
            fields[key] = match.group(1).strip()
    wl = re.search(r"(?is)^\s*wavelength\s*=\s*\{([^}]+)\}", raw, flags=re.MULTILINE)
    if wl:
        fields["wavelengths"] = np.array([float(n) for n in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", wl.group(1))], dtype=np.float64)
    required = ("samples", "lines", "bands", "data type", "interleave")
    if any(key not in fields for key in required):
        raise ValueError(f"Incomplete ENVI HDR: {path}")
    return fields


def envi_memmap(hdr_path: Path, raw_path: Path):
    meta = read_envi_header(hdr_path)
    dtypes = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    code = int(meta["data type"])
    if code not in dtypes:
        raise ValueError(f"Unsupported ENVI datatype {code}: {hdr_path}")
    order = "<" if int(meta.get("byte order", 0)) == 0 else ">"
    dtype = np.dtype(dtypes[code]).newbyteorder(order)
    h, w, b = (int(meta[k]) for k in ("lines", "samples", "bands"))
    interleave = meta["interleave"].strip().lower()
    shapes = {"bil": (h, b, w), "bsq": (b, h, w), "bip": (h, w, b)}
    if interleave not in shapes:
        raise ValueError(f"Unknown interleave {interleave}")
    offset = int(meta.get("header offset", 0))
    required_bytes = offset + h * w * b * dtype.itemsize
    if raw_path.stat().st_size < required_bytes:
        raise ValueError(f"RAW too small: {raw_path} ({raw_path.stat().st_size} < {required_bytes})")
    cube = np.memmap(raw_path, mode="r", dtype=dtype, offset=offset, shape=shapes[interleave])
    return cube, meta


def read_patch(cube, meta: dict, y: int, x: int, size: int) -> np.ndarray:
    mode = meta["interleave"].strip().lower()
    if mode == "bil":
        block = cube[y:y + size, :, x:x + size].transpose(0, 2, 1)
    elif mode == "bsq":
        block = cube[:, y:y + size, x:x + size].transpose(1, 2, 0)
    else:
        block = cube[y:y + size, x:x + size, :]
    return np.asarray(block, dtype=np.float32)


def discover_case_files(class_dir: Path, diagnostic_label: str) -> list[dict]:
    if not class_dir.exists():
        raise FileNotFoundError(class_dir)
    cases = []
    for hdr in sorted(class_dir.glob("*.hdr"), key=lambda p: p.name.casefold()):
        raw_options = [p for p in hdr.parent.iterdir() if p.stem.casefold() == hdr.stem.casefold() and p.suffix.casefold() == ".raw"]
        if len(raw_options) != 1:
            raise FileNotFoundError(f"Expected exactly one matching RAW for {hdr.name}, found {len(raw_options)}")
        # _ref is the calibrated patient sample, not a separate patient ID.
        case_id = re.sub(r"(?i)_ref$", "", hdr.stem)
        cases.append({"patient_id": f"{diagnostic_label}_{case_id}",
                      "case_id": case_id, "label": LABELS[diagnostic_label],
                      "class": diagnostic_label, "hdr_path": str(hdr), "raw_path": str(raw_options[0])})
    ids = [x["patient_id"] for x in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate patient IDs in {class_dir}; check multiple scans per case")
    return cases


def build_manifest() -> Path:
    """Create a frozen manifest from the two class folders if none exists."""
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = CFG.output_dir / "patient_manifest.csv"
    if manifest_path.exists():
        print(f"Existing patient manifest kept unchanged: {manifest_path}")
        return manifest_path

    ehcc_cases = discover_case_files(CFG.data_root / CFG.ehcc_folder, "ehcc")
    hgdn_cases = discover_case_files(CFG.data_root / CFG.hgdn_folder, "hgdn")
    if len(ehcc_cases) != CFG.selected_ehcc or len(hgdn_cases) != CFG.selected_hgdn:
        raise RuntimeError(
            f"Expected {CFG.selected_ehcc} eHCC and {CFG.selected_hgdn} HGDN cases; "
            f"found {len(ehcc_cases)} and {len(hgdn_cases)}. Check the cohort folders."
        )
    records = ehcc_cases + hgdn_cases
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "case_id", "label", "class", "hdr_path", "raw_path"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Created frozen 60-patient manifest: {manifest_path}")
    return manifest_path


def load_manifest(path: Path | None = None) -> list[dict]:
    path = path or CFG.output_dir / "patient_manifest.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        result = list(csv.DictReader(stream))
    for item in result:
        item["label"] = int(item["label"])
    return result


def wavelength_indices(meta: dict) -> list[int]:
    wave = meta.get("wavelengths")
    if wave is None or len(wave) != int(meta["bands"]):
        raise ValueError("Missing or inconsistent ENVI wavelength metadata")
    indices = [int(np.argmin(np.abs(wave - wavelength))) for wavelength in CFG.rgb_wavelengths_nm]
    for nominal, idx in zip(CFG.rgb_wavelengths_nm, indices):
        if abs(float(wave[idx]) - nominal) > CFG.max_rgb_band_distance_nm:
            raise ValueError(f"Nominal RGB {nominal}nm, nearest {wave[idx]}nm exceeds tolerance")
    return indices


def make_dual_inputs(patch: np.ndarray, band_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return pseudo-RGB and 25 regional mean/std spectra [25,2,bands]."""
    h, w, b = patch.shape
    g = CFG.grid_size
    if h != CFG.patch_size or w != CFG.patch_size or h % g != 0 or w % g != 0:
        raise ValueError(f"Patch shape {patch.shape} incompatible with {g}x{g} regional grid")
    finite_patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(finite_patch[..., band_ids], CFG.rgb_clip_min, CFG.rgb_clip_max)
    rgb = (rgb - CFG.rgb_clip_min) / (CFG.rgb_clip_max - CFG.rgb_clip_min)
    rgb = rgb.transpose(2, 0, 1).astype(np.float16)

    # Match Experiment C exactly: split 100x100 into 25 non-overlapping 20x20 regions.
    grouped = finite_patch.reshape(g, h // g, g, w // g, b).transpose(0, 2, 1, 3, 4)
    means = grouped.mean(axis=(2, 3)).astype(np.float32)  # [5,5,bands]
    stds = grouped.std(axis=(2, 3)).astype(np.float32)   # [5,5,bands]
    if CFG.use_region_snv:
        centering = means.mean(axis=-1, keepdims=True)
        scaling = np.maximum(means.std(axis=-1, keepdims=True), CFG.eps)
        means = (means - centering) / scaling
        stds = stds / scaling
    spectral = np.stack((means, stds), axis=2).reshape(g * g, 2, b).astype(np.float32)
    return rgb, spectral


def prepare_cache() -> Path:
    records = load_manifest()
    CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = CFG.output_dir / f"patch_index_region25_seed{CFG.seed}.csv"
    patch_rows = []
    for i, record in enumerate(records, 1):
        case = record["patient_id"]
        rgb_file = CFG.cache_dir / f"{case}_rgb_region25_v1.npy"
        spec_file = CFG.cache_dir / f"{case}_spec_region25_mean_std_v1.npy"
        cube, meta = envi_memmap(Path(record["hdr_path"]), Path(record["raw_path"]))
        h, w = int(meta["lines"]), int(meta["samples"])
        n_y, n_x = h // CFG.patch_size, w // CFG.patch_size
        num = n_y * n_x
        ids = wavelength_indices(meta)
        print(f"[{i}/{len(records)}] {case}, {h}x{w}x{meta['bands']}, {num} patches, RGB bands {ids}, 5x5 regional mean/std input")
        expected_rgb_shape = (num, 3, CFG.patch_size, CFG.patch_size)
        expected_spec_shape = (num, CFG.grid_size ** 2, 2, int(meta["bands"]))
        cache_ok = False
        if rgb_file.exists() and spec_file.exists():
            try:
                cache_ok = (np.load(rgb_file, mmap_mode="r").shape == expected_rgb_shape and
                            np.load(spec_file, mmap_mode="r").shape == expected_spec_shape)
            except Exception:
                cache_ok = False
        if not cache_ok:
            # Versioned cache prevents accidental reuse of the previous regional-spectrum format.
            rgbs = np.lib.format.open_memmap(rgb_file, mode="w+", dtype="float16", shape=expected_rgb_shape)
            specs = np.lib.format.open_memmap(spec_file, mode="w+", dtype="float32", shape=expected_spec_shape)
            for y_idx in range(n_y):
                for x_idx in range(n_x):
                    j = y_idx * n_x + x_idx
                    block = read_patch(cube, meta, y_idx * CFG.patch_size, x_idx * CFG.patch_size, CFG.patch_size)
                    rgbs[j], specs[j] = make_dual_inputs(block, ids)
            rgbs.flush()
            specs.flush()
            del rgbs, specs
        for y_idx in range(n_y):
            for x_idx in range(n_x):
                patch_rows.append({"patient_id": case, "label": record["label"],
                                   "patch_index": y_idx * n_x + x_idx,
                                   "y": y_idx * CFG.patch_size, "x": x_idx * CFG.patch_size,
                                   "rgb_file": str(rgb_file), "spec_file": str(spec_file)})
        del cube
    with index_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(patch_rows[0]))
        writer.writeheader()
        writer.writerows(patch_rows)
    info = {"positive_class": "eHCC", "label_mapping": LABELS, "patch_size": CFG.patch_size,
            "spectral_input": "5x5 non-overlapping regional mean and standard deviation; shared regional SNV",
            "use_region_snv": CFG.use_region_snv,
            "rgb_wavelengths_nm": CFG.rgb_wavelengths_nm,
            "cache_version": "region25_mean_std_v1",
            "note": "uses the verified frozen patient manifest; source RAW cubes are read-only"}
    (CFG.output_dir / f"preprocess_metadata_region25_seed{CFG.seed}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {len(patch_rows)} patches to {index_path}")
    return index_path


class DualHSIDataset(Dataset):
    """Cache-backed loader; choose patient IDs BEFORE creating datasets/splits."""
    def __init__(self, patch_index: Path, patient_ids: set[str], train: bool = False):
        with Path(patch_index).open("r", encoding="utf-8-sig", newline="") as stream:
            self.items = [r for r in csv.DictReader(stream) if r["patient_id"] in patient_ids]
        self.train = train
        self._cache: dict[str, np.ndarray] = {}
        if not self.items:
            raise ValueError("No patches match the patient ID selection")

    def __len__(self):
        return len(self.items)

    def _read(self, filename: str, index: int):
        if filename not in self._cache:
            self._cache[filename] = np.load(filename, mmap_mode="r")
        return np.array(self._cache[filename][index], dtype=np.float32, copy=True)

    def __getitem__(self, index: int):
        r = self.items[index]
        j = int(r["patch_index"])
        rgb = self._read(r["rgb_file"], j)
        spec = self._read(r["spec_file"], j)
        if spec.shape != (CFG.grid_size ** 2, 2, 256):
            raise RuntimeError(f"Expected regional spectrum (25,2,256), got {spec.shape}; rerun dataset.py --prepare")
        if self.train:
            # Apply identical spatial transforms to RGB and the aligned 5x5 spectral grid.
            spec_grid = spec.reshape(CFG.grid_size, CFG.grid_size, 2, 256)
            if np.random.random() < 0.5:
                rgb = np.flip(rgb, axis=2)
                spec_grid = np.flip(spec_grid, axis=1)
            if np.random.random() < 0.5:
                rgb = np.flip(rgb, axis=1)
                spec_grid = np.flip(spec_grid, axis=0)
            k = int(np.random.randint(0, 4))
            rgb = np.rot90(rgb, k, axes=(1, 2))
            spec_grid = np.rot90(spec_grid, k, axes=(0, 1))
            spec = spec_grid.reshape(CFG.grid_size ** 2, 2, 256)
        return (torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(np.ascontiguousarray(spec)),
                int(r["label"]), r["patient_id"], j)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="create and freeze the 30+30 patient manifest from the class folders")
    parser.add_argument("--prepare", action="store_true", help="cache patches from calibrated ENVI data")
    args = parser.parse_args()
    if args.scan:
        build_manifest()
    if args.prepare:
        manifest = CFG.output_dir / "patient_manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Verified V3+ patient_manifest.csv is required and was not found: {manifest}. Do not create a new cohort automatically.")
        records = load_manifest(manifest)
        if len(records) != 60 or sum(int(r["label"]) == 1 for r in records) != 30:
            raise RuntimeError("Expected the exact frozen 60-patient V3+ cohort (30 eHCC + 30 HGDN).")
        prepare_cache()
    if not args.scan and not args.prepare:
        parser.print_help()


if __name__ == "__main__":
    main()
