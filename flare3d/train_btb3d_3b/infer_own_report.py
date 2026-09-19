#!/usr/bin/env python
"""Report generation with our own trained model (BTB3D-8 encoder + Llama-3.2-3B +
Stage-1 projector + report LoRA) on the FLARE26 3D validation set.

Model loading, the two monkeypatches, and the image-token convention are shared with
`infer_own_vqa.py` -- imported rather than copied so the '<image>' fix and its guard
assert cannot drift between the two inference paths.

The prompt is exactly what `build_report_split.py` wrote into report_train.json, so
this asks the model for the format it was trained to produce. Output is written in the
`generated,gt,name` schema the baselines' report CSVs use, so score_green.py and
score_crimson.py read it unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_own_vqa import apply_patches, load_model          # noqa: E402
from report_format import REPORT_PROMPT, normalize_report    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base",
                    default=os.environ.get("LLM_PATH", "meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--report-csv", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    # The necessity test, for the report side. It has been run on VQA -- with the visual
    # tokens zeroed the top-1 answer token is unchanged 98.9% of the time -- but never on
    # report generation, and our reports are 19 templated region lines per case. If the
    # text is the same without the image, GREEN is scoring a template and every report
    # experiment so far has been measuring one.
    ap.add_argument("--image-mode", choices=["real", "zero"], default="real")
    a = ap.parse_args()

    apply_patches()
    device = "cuda"
    tokenizer, model = load_model(a.model_base, a.checkpoint, device)

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    cases = json.loads(open(a.val_json).read())
    if a.limit:
        cases = cases[: a.limit]

    rows = []
    for sample in tqdm(cases):
        case_id = sample["case_id"]
        stem = case_id.replace(".npy", "").replace(".nii.gz", "")
        enc_path = os.path.join(a.enc_dir, f"{stem}.nii_embedded.npz")
        gt = "\n".join(f"{k}: {v}" for k, v in sample["findings"].items())
        if not os.path.exists(enc_path):
            print(f"[warn] missing encoding for {stem}", flush=True)
            continue

        arr = np.load(enc_path)["arr"].transpose(0, 2, 3, 4, 1)  # (1,T,H,W,C)
        if a.image_mode == "zero":
            arr = np.zeros_like(arr)
        image_tensor = torch.tensor(arr).to(device, dtype=torch.bfloat16)

        conv = conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{REPORT_PROMPT}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)
        n_slots = int((input_ids == IMAGE_TOKEN_INDEX).sum())
        assert n_slots == 1, f"expected exactly 1 image slot in the prompt, got {n_slots}"

        try:
            with torch.inference_mode():
                out = model.generate(input_ids, images=image_tensor,
                                     max_new_tokens=a.max_new_tokens,
                                     do_sample=False, use_cache=True)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = normalize_report(
                text.replace(prompt.replace("<image>", "").strip(), "").strip())
        except Exception as e:
            print(f"  ERROR {stem}: {type(e).__name__}: {e}", flush=True)
            pred = ""

        rows.append(dict(generated=pred, gt=gt, name=stem))
        del image_tensor
        pd.DataFrame(rows).to_csv(a.report_csv, index=False)

    print(f"saved -> {a.report_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
