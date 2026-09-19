"""Replace follow-up answers that fall outside the offered choice list.

Measured on the submitted model over the 681 validation cases: 35.1 % of follow-up
predictions (1,168 of 3,323) lie outside the choice list, and 98.9 % of those are the
literal string "none of the above" -- an option the training prompt appends to every
choice list but which never appears as a reference answer. Because the reference answer
is always one of the offered choices, every out-of-list prediction is an automatic zero,
so replacing them cannot lose points.

The replacement rule is the answer-position distribution of the training split, estimated
once and applied unchanged.
"""
from __future__ import annotations
import argparse, collections, json, math, os
from pathlib import Path
import pandas as pd

def train_files():
    """Training-split JSONs under $FLARE_TRAIN_DIR. The chest file is called `CT-RATE-Tr.json` in the
    dataset README but `CT-RATE-Tr .json` (with a space) in our copy, so both spellings are tried."""
    d = os.environ.get("FLARE_TRAIN_DIR")
    if not d:
        raise SystemExit("set FLARE_TRAIN_DIR to the dataset's train/ directory, or pass --prior "
                         "configs/offlist_prior.json to use the precomputed table")
    files = [Path(d) / "CT-AMOS-Tr.json"]
    files.append(next(p for p in (Path(d) / "CT-RATE-Tr.json", Path(d) / "CT-RATE-Tr .json") if p.exists()))
    return files


def canon(c):
    f = Path(str(c)).name
    return f[:-7] if f.endswith(".nii.gz") else Path(f).stem


def nrm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip().lower()


NOTA = "none of the above"


def train_position_prior():
    """Position of the reference answer within its choice list, over the training split,
    after removing "None of the above".

    Removing it first is essential. 48.8 % of training answers sit in the last position,
    and that last entry is "None of the above" itself. Since that option never appears as
    a reference answer at evaluation time, a prior computed with it left in would report
    "last position is most common" and, transferred to evaluation, would select the rarest
    position instead.
    """
    by_n = collections.defaultdict(collections.Counter)
    for f in train_files():
        for d in json.load(open(f)):
            for q in d["local_vqa"]:
                ch = q.get("choices")
                if q["follow_up"] == -1 or not ch:
                    continue
                keep = [c for c in ch if nrm(c) != NOTA]
                if nrm(q["answer"]) == NOTA:
                    continue
                try:
                    by_n[len(keep)][keep.index(q["answer"])] += 1
                except ValueError:
                    pass
    return {n: c.most_common(1)[0][0] for n, c in by_n.items()}, by_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="preds/rag_calib_prob.csv")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--val-json", default="preproc/val_processed.json")
    ap.add_argument("--prior", default=None,
                    help="precomputed position table (configs/offlist_prior.json); when given the "
                         "training files are not read")
    a = ap.parse_args()
    import sys; sys.path.insert(0, ".")
    from score_vqa import build_chains

    if a.prior:
        pos = {int(k): int(v) for k, v in json.load(open(a.prior))["position"].items()}
        dist = {}
    else:
        pos, dist = train_position_prior()
    for n, c in sorted(dist.items()):
        tot = sum(c.values())
        print(f"choice lists of {n}: n={tot}  position distribution {dict(c.most_common(4))}  "
              f"modal position {pos[n]} ({c[pos[n]]/tot:.3f})")

    gt = {canon(d["case_id"]): d for d in json.load(open(a.val_json))}
    df = pd.read_csv(a.pred_csv)
    loc = df[df.scope == "local"].copy()
    fixed, seen = 0, 0
    newp = []
    for _, r in loc.iterrows():
        cid = canon(r.case_id)
        g = gt.get(cid)
        p = [x.strip() for x in str(r.prediction).split("|")]
        if g is not None:
            ch = build_chains(g["local_vqa"])
            byid = {q["id"]: q for q in g["local_vqa"]}
            root = int(r.question_id)
            if root in ch:
                for k, qid in enumerate(ch[root][1:], 1):
                    if k >= len(p):
                        break
                    opts = byid[qid].get("choices") or []
                    if not opts:
                        continue
                    seen += 1
                    if nrm(p[k]) not in [nrm(o) for o in opts]:
                        p[k] = opts[min(pos.get(len(opts), 0), len(opts) - 1)]
                        fixed += 1
        newp.append("|".join(p))
    loc["prediction"] = newp
    pd.concat([df[df.scope == "global"], loc], ignore_index=True).to_csv(a.out_csv, index=False)
    print(f"\nreplaced {fixed} of {seen} follow-up answers ({fixed/max(seen,1):.1%})  -> {a.out_csv}")


if __name__ == "__main__":
    main()
