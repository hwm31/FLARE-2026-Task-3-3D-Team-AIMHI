"""Preprocess the FLARE26 3D validation volumes into the two baseline .npy formats.

Faithful to the upstream scripts, which are identical apart from the resize target:
  - Med3DVLM  `flare_preprocess.py`            -> (1, 128, 256, 256)
  - Phi3/LaMed `Data/process/process_ct.py`    -> (1,  32, 256, 256)

Reading + gunzipping the NIfTI is the expensive part, so we do it once per case and
emit both targets. Volumes are cast to float32 right after load to keep worker RSS
bounded (CT-RATE volumes are float64 and can exceed 1.9 GB each); the clip/min-max/
resize math is otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm

TARGETS = {"med3dvlm": (128, 256, 256), "lamed": (32, 256, 256)}


def _process_case(args):
    case, nifti_root, out_roots = args
    case_id = case["case_id"]
    nifti_p = Path(nifti_root) / case_id
    stub = case_id.replace(".nii.gz", "")

    todo = {k: Path(r) / f"{stub}.npy" for k, r in out_roots.items()}
    todo = {k: p for k, p in todo.items() if not p.exists()}

    if todo:
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(nifti_p)))
        if vol.ndim == 4:
            vol = vol[0]
        vol = vol.astype(np.float32, copy=False)

        # upstream: AMOS gets an abdominal window, CT-RATE a lung window
        if "amos" in str(nifti_p):
            vol = np.clip(vol, -160.0, 240.0)
        else:
            vol = np.clip(vol, -1350.0, 150.0)
        vol = (vol - vol.min()) / (vol.max() - vol.min())

        for key, out_p in todo.items():
            r = resize(vol, TARGETS[key], anti_aliasing=True).astype(np.float32)[None]
            # np.save appends '.npy' unless the name already ends in it
            tmp = out_p.with_name(f"{out_p.stem}.tmp.npy")
            np.save(tmp, r)
            os.replace(tmp, out_p)
            del r
        del vol

    new_case = OrderedDict(case)
    new_case["case_id"] = f"{stub}.npy"
    return new_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_in", type=Path, required=True)
    ap.add_argument("--nifti_dir", type=Path, required=True)
    ap.add_argument("--out_base", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()

    out_roots = {k: str(a.out_base / k) for k in TARGETS}
    for r in out_roots.values():
        Path(r).mkdir(parents=True, exist_ok=True)

    cases = json.loads(a.json_in.read_text(), object_pairs_hook=OrderedDict)
    payload = [(c, str(a.nifti_dir), out_roots) for c in cases]

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        new_cases = list(
            tqdm(pool.map(_process_case, payload, chunksize=1),
                 total=len(cases), desc="volumes")
        )

    out_json = a.out_base / f"{a.json_in.stem}_processed.json"
    out_json.write_text(json.dumps(new_cases, indent=2))
    print(f"volumes -> {a.out_base}/{{{','.join(TARGETS)}}}")
    print(f"json    -> {out_json}")


if __name__ == "__main__":
    main()
