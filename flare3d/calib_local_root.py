"""Decide each local chain's ROOT answer by thresholding P(yes) instead of by generation.

local decomposes as 0.617 x 0.837: the follow-ups are already 84% right and the whole axis
is gated on a yes/no root the model gets 61.7% of -- exactly what the best per-chain-length
constant answer scores (61.6%). So on the root question the model contributes nothing, and
it is badly miscalibrated: it says yes 73% of the time against a 55% base rate, with the
bias reversing by chain length (88.8% yes on length-2 chains against 56.7% true; 17.1% on
length-3 against 36.1%).

Generation cannot be recalibrated -- it emits one token. A threshold can, and the same move
on chest global was worth +0.055.

Thresholds are fitted on OUR-VAL only, per chain length. Fitting on FLARE val would be
reading the test set; the number that comes out of that is not a prediction of anything.
Only the root answer changes -- follow-ups are copied through, because a "no" root does NOT
imply "None of the above" follow-ups in this data (checked: 0 of 1,727).
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from score_vqa import canonical, norm, build_chains


def load_gt(p):
    raw = json.load(open(p))
    return {canonical(r["case_id"]): r for r in raw} if isinstance(raw, list) \
        else {canonical(k): v for k, v in raw.items()}


def roots(gt, root_json):
    """(chain_len, p_yes, gt_root_answer) for every chain the probe covered."""
    R = {canonical(k): v for k, v in json.load(open(root_json)).items()}
    out = []
    for cid, v in gt.items():
        if cid not in R:
            continue
        ch = build_chains(v["local_vqa"])
        ans = {q["id"]: norm(q["answer"]) for q in v["local_vqa"]}
        for root, ids in ch.items():
            if str(root) in R[cid]:
                out.append((len(ids), R[cid][str(root)]["p_yes"], ans[root]))
    return out


def fit(rows, grid=np.arange(0.02, 0.99, 0.01)):
    thr = {}
    for L in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == L]
        acc, t = max(((np.mean([("yes" if p >= t else "no") == g for _, p, g in sub]), t)
                      for t in grid), key=lambda x: x[0])
        thr[L] = float(t)
    return thr


def rescore(pred_csv, gt, root_json, thr, out_csv=None):
    R = {canonical(k): v for k, v in json.load(open(root_json)).items()}
    d = pd.read_csv(pred_csv)
    loc = d[d.scope == "local"].copy()
    tot = n = 0
    newpred = []
    for i, r in loc.iterrows():
        cid = canonical(r.case_id)
        ch = build_chains(gt[cid]["local_vqa"])
        ans = {q["id"]: norm(q["answer"]) for q in gt[cid]["local_vqa"]}
        root = int(r.question_id)
        p = [x.strip() for x in str(r.prediction).split("|")]
        if root in ch and cid in R and str(root) in R[cid]:
            L = len(ch[root])
            p = list(p)
            p[0] = "Yes" if R[cid][str(root)]["p_yes"] >= thr.get(L, 0.5) else "No"
        newpred.append(" | ".join(p))
        if root not in ch:
            continue
        n += 1
        ids = ch[root]
        if norm(p[0]) != ans[root]:
            continue
        tot += sum(1 for k, q in enumerate(ids)
                   if (norm(p[k]) if k < len(p) else "") == ans[q]) / len(ids)
    loc["prediction"] = newpred
    if out_csv:
        pd.concat([d[d.scope == "global"], loc], ignore_index=True).to_csv(out_csv, index=False)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred-csv", default=None)
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()
    ov_gt, fv_gt = load_gt("preproc/our_val_processed.json"), load_gt("preproc/val_processed.json")
    ov = roots(ov_gt, f"results/root_ov_{a.run}.json")
    fv = roots(fv_gt, f"results/root_fv_{a.run}.json")
    thr = fit(ov)
    print(f"{a.run}: held-out chains {len(ov)}  validation chains {len(fv)}")
    for L in sorted(thr):
        so, sf = [r for r in ov if r[0] == L], [r for r in fv if r[0] == L]
        ao = np.mean([("yes" if p >= thr[L] else "no") == g for _, p, g in so])
        af = np.mean([("yes" if p >= thr[L] else "no") == g for _, p, g in sf])
        cf = max(np.mean([g == "yes" for _, _, g in sf]), np.mean([g == "no" for _, _, g in sf]))
        print(f"  length {L}: threshold {thr[L]:.2f}  held-out root acc {ao:.3f}  "
              f"validation {af:.3f}  (constant-answer baseline {cf:.3f})")
    pred = a.pred_csv or f"preds/{a.run}_prob.csv"
    base = rescore(pred, fv_gt, f"results/root_fv_{a.run}.json", {}, None)
    print(f"\n  local, generation as-is : confirm with score_vqa")
    new = rescore(pred, fv_gt, f"results/root_fv_{a.run}.json", thr, a.out_csv)
    print(f"  local after calibration : {new:.4f}")
    if a.out_csv:
        print(f"  saved -> {a.out_csv}")


if __name__ == "__main__":
    main()
