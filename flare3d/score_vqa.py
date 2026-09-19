"""Global / local VQA accuracy for the FLARE26 3D task.

The scoring functions are taken verbatim from the baselines' own scorers -- Med3DVLM's
`src/flare_infer/eval_vqa.py` and FLARE25-MLLM3D's `eval_vqa.py` are byte-identical in
logic, so one implementation keeps every model comparable. Only the CLI and the extra
diagnostic counters are ours.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def canonical(cid: str) -> str:
    fname = Path(str(cid)).name
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    return Path(fname).stem


def norm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip().lower()


def global_accuracy(df: pd.DataFrame, gt: dict) -> tuple[float, int]:
    glob = df[df.scope == "global"]
    tot, n = 0.0, 0
    for _, r in glob.iterrows():
        cid = canonical(r.case_id)
        preds = [norm(p) for p in str(r.prediction).split(",") if norm(p)]
        gts = [norm(a) for a in gt[cid]["global_vqa"][0]["answer"]]
        if not gts and not preds:
            acc = 1.0
        else:
            acc = sum(p in gts for p in preds) / max(len(preds), len(gts))
        tot += acc
        n += 1
    return (0.0 if n == 0 else tot / n), n


def build_chains(local_gt):
    by_id = {q["id"]: q for q in local_gt}
    chains = {}
    for q in local_gt:
        root = q["id"]
        while by_id[root]["follow_up"] != -1:
            root = by_id[root]["follow_up"]
        chains.setdefault(root, set()).add(q["id"])
    return {r: [r] + sorted(ids - {r}) for r, ids in chains.items()}


def local_accuracy(df: pd.DataFrame, gt: dict) -> tuple[float, int]:
    loc = df[df.scope == "local"]
    chain_score_sum, chain_cnt = 0.0, 0

    for cid_csv, rows in loc.groupby("case_id"):
        cid = canonical(cid_csv)
        local_gt = gt[cid]["local_vqa"]
        chains = build_chains(local_gt)
        gt_answers = {q["id"]: norm(q["answer"]) for q in local_gt}

        for _, r in rows.iterrows():
            root_id = int(r.question_id)
            if root_id not in chains:
                continue
            preds_list = [norm(p) for p in str(r.prediction).split("|")]
            qids = chains[root_id]
            chain_cnt += 1
            if not preds_list or preds_list[0] != gt_answers[root_id]:
                continue
            correct = sum(
                1 for idx, qid in enumerate(qids)
                if (preds_list[idx] if idx < len(preds_list) else "") == gt_answers[qid]
            )
            chain_score_sum += correct / len(qids)

    return (0.0 if chain_cnt == 0 else chain_score_sum / chain_cnt), chain_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--val_json", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    a = ap.parse_args()

    preds = pd.read_csv(a.pred_csv)
    gt = {canonical(d["case_id"]): d for d in json.loads(a.val_json.read_text())}

    g, n_g = global_accuracy(preds, gt)
    l, n_l = local_accuracy(preds, gt)

    out = {
        "global_accuracy": round(g, 6),
        "local_accuracy": round(l, 6),
        "n_global_questions": n_g,
        "n_local_chains": n_l,
        "n_rows": len(preds),
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
