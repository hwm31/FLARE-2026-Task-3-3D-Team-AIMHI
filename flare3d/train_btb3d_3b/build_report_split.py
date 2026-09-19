"""Build report-generation instructions from the SAME clean split as the VQA ones.

The per-task LoRA design needs a report adapter trained on exactly the cases that the
VQA adapter trained on -- otherwise the two adapters would disagree about what counts
as held-out, and our_val would stop being a valid comparison point across tasks.
So this reads split_manifest.json rather than re-splitting.

Prompt is the 18-region format the shared GREEN scorer parses (one 'region: findings'
line per region), matching report_format.REPORT_PROMPT used by the zero-shot adapters,
so a trained model's output is scored by exactly the same parser as the baselines'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_format import REPORT_PROMPT, REGIONS  # noqa: E402

DEFAULT_IMAGE_TOKEN = "<image>"  # llava.constants, inlined to keep this a plain data script

# --fill-normal writes a normal statement for regions the reference never mentions.
# Two variants on purpose, because they answer different questions:
#   natural  -- the prompt asks for all 18 regions, so answering all 18 is following the
#               instruction. Measures "does full coverage help?"
#   template -- the SAME text score_green.py uses as the hypothesis for a region the
#               reference lacks ("<region> is normal."), so those pairs become a near
#               string match. Measures "how much of the gain is exploiting the scorer?"
# Running only `template` would conflate the two and overstate a real improvement.
FILL = {
    "natural":  lambda r: f"No abnormality is identified in the {r}.",
    "template": lambda r: f"{r} is normal.",
}

import os

if not os.environ.get("FLARE_TRAIN_DIR"):
    raise SystemExit("set FLARE_TRAIN_DIR to the dataset's train/ directory")
FLARE_TRAIN = Path(os.environ["FLARE_TRAIN_DIR"])
# `CT-RATE-Tr.json` in the dataset README, `CT-RATE-Tr .json` (with a space) in our copy
CT_RATE_JSON = next(p for p in (FLARE_TRAIN / "CT-RATE-Tr.json", FLARE_TRAIN / "CT-RATE-Tr .json") if p.exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fill-normal", choices=["off", "natural", "template"], default="off")
    ap.add_argument("--suffix", default="", help="appended to the output filenames")
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    cases = {c["case_id"]: c for c in
             json.loads((FLARE_TRAIN / "CT-AMOS-Tr.json").read_text())
             + json.loads(CT_RATE_JSON.read_text())}

    def build(ids):
        out = []
        for cid in ids:
            c = cases[cid]
            if not c.get("findings"):
                continue
            found = c["findings"]
            if a.fill_normal == "off":
                gt = "\n".join(f"{k}: {v}" for k, v in found.items())
            else:
                fill = FILL[a.fill_normal]
                # canonical order so the model learns a fixed layout; regions the
                # reference does mention keep their real text
                gt = "\n".join(f"{r}: {found.get(r) or fill(r)}" for r in REGIONS)
            out.append({"id": f"report_{cid}", "image": cid,
                        "conversations": [
                            # REPORT_PROMPT is the zero-shot adapters' prompt, where the
                            # image is passed out of band. LLaVA training needs the
                            # literal "<image>" in the text: preprocess_multimodal only
                            # acts on sentences containing it, and without it
                            # prepare_inputs_labels_for_multimodal takes its
                            # num_images==0 branch and splices in an EMPTY image slice,
                            # so the run trains on text alone with no error raised.
                            {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{REPORT_PROMPT}"},
                            {"from": "gpt", "value": gt}]})
        return out

    tr = build(man["our_train"]["case_ids"])
    va = build(man["our_val"]["case_ids"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sfx = a.suffix
    (a.out_dir / f"report{sfx}_train.json").write_text(json.dumps(tr, indent=1))
    (a.out_dir / f"report{sfx}_val.json").write_text(json.dumps(va, indent=1))

    def dom(ids):
        am = sum(1 for i in ids if i.startswith("amos"))
        return f"amos {am} + ctrate {len(ids)-am}"
    print(f"report_train : {len(tr):5d}  ({dom([x['image'] for x in tr])})")
    print(f"report_val   : {len(va):5d}  ({dom([x['image'] for x in va])})")
    print(f"saved -> {a.out_dir}")


if __name__ == "__main__":
    main()
