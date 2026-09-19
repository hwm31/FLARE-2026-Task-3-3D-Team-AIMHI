"""Apply pre-fitted post-processing rules as-is (the path the submission container takes).

calib_local_root.py *refits* the thresholds on the internal held-out split. The container
has no access to that split, so the deployed path cannot do the same. This script instead
reads the thresholds baked into configs/root_thresholds.json and only applies them.
Verifying the released bundle has to go through this path to measure what the container
actually does.
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from score_vqa import canonical, norm, build_chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--root-json", required=True, help="root P(yes) for the target split")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--val-json", default="preproc/val_processed.json")
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    thr = {int(k): float(v) for k, v in json.load(open(a.thresholds)).items()}
    R = {canonical(k): v for k, v in json.load(open(a.root_json)).items()}
    gt = {canonical(d["case_id"]): d for d in json.load(open(a.val_json))}
    df = pd.read_csv(a.pred_csv)
    n = 0
    for i, r in df.iterrows():
        if r.get("scope") != "local":
            continue
        cid = canonical(str(r["case_id"]))
        if cid not in R or cid not in gt:
            continue
        ch = build_chains(gt[cid]["local_vqa"])
        root = int(r["question_id"])
        if root not in ch or str(root) not in R[cid]:
            continue
        p = R[cid][str(root)]["p_yes"]
        t = thr.get(len(ch[root]))
        if t is None:
            continue
        parts = str(r["prediction"]).split("|")
        parts[0] = "Yes" if p >= t else "No"
        df.at[i, "prediction"] = "|".join(parts)
        n += 1
    df.to_csv(a.out_csv, index=False)
    print(f"re-decided {n} root answers -> {a.out_csv}")


if __name__ == "__main__":
    main()
