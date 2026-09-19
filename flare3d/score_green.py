"""Region-wise GREEN scoring for the FLARE26 3D report-generation task.

Semantics are copied from the baselines' `generate_green_score.py` (identical in the
Med3DVLM and FLARE25-MLLM3D repos):

  * predictions and ground truth are parsed into {region: text} dicts;
  * a region present in BOTH is scored GREEN(ref=gt, hyp=pred);
  * a region the model invented (in pred, not in gt) is scored against "<region> is
    normal." using the upstream argument order, i.e. refs=[pred], hyps=["... normal."];
  * a region the model omitted (in gt, not in pred) scores 0.0 -- parse failures are
    therefore 0, never dropped, which is also what the organisers do;
  * per-case green = mean over that case's region scores;
  * summary green = mean over cases; region_means = mean over cases where the region
    appears in the GROUND TRUTH.

The only departure from upstream is batching: upstream calls the 7B judge once per
case (~18 pairs), which wastes most of the GPU. We collect every (ref, hyp) pair across
all cases, score them in one pass, and scatter the results back. GREEN returns a list
aligned to its inputs, so the arithmetic is unchanged -- only the batch shape differs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def to_dict(text) -> dict:
    """Upstream parser: JSON object, else 'region: text' lines."""
    if isinstance(text, dict):
        return text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    out = {}
    for line in filter(None, map(str.strip, str(text).splitlines())):
        m = re.match(r"([^:–\-]+)[:–\-]\s*(.+)", line)
        if m:
            key, val = m.groups()
            out[key.strip().lower()] = val.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_csv", type=Path, required=True,
                    help="CSV with 'gt' and 'generated' columns")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--model", default="StanfordAIMI/GREEN-RadLlama2-7b")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.report_csv).fillna("")
    if a.limit:
        df = df.head(a.limit)

    # ---- build the full pair list ----------------------------------------
    refs, hyps = [], []
    slots = []          # (case_idx, region, pair_index) for pairs we actually score
    zero_regions = []   # (case_idx, region) omissions -> 0.0

    for i, row in df.iterrows():
        gt_d, gen_d = to_dict(row["gt"]), to_dict(row["generated"])
        for r in [r for r in gen_d if r in gt_d]:
            slots.append((i, r, len(refs)))
            refs.append(gt_d[r]); hyps.append(gen_d[r])
        for r in gen_d.keys() - gt_d.keys():
            slots.append((i, r, len(refs)))
            refs.append(gen_d[r]); hyps.append(f"{r} is normal.")
        for r in gt_d.keys() - gen_d.keys():
            zero_regions.append((i, r))

    n_parsed = sum(1 for _, row in df.iterrows() if to_dict(row["generated"]))
    print(f"cases={len(df)}  parsed_into_regions={n_parsed} "
          f"({n_parsed / max(len(df), 1):.3f})  pairs_to_score={len(refs)}  "
          f"omitted_regions={len(zero_regions)}", flush=True)

    green_list = []
    if refs:
        from green_score import GREEN
        scorer = GREEN(model_name=a.model, output_dir=".")  # no installed green_score build takes cache_dir
        scorer.batch_size = a.batch_size
        _, _, green_list, *_ = scorer(refs=refs, hyps=hyps)

    # ---- scatter back -----------------------------------------------------
    per_case: list[dict] = [dict() for _ in range(len(df))]
    for case_i, region, pair_i in slots:
        per_case[case_i][region] = float(green_list[pair_i])
    for case_i, region in zero_regions:
        per_case[case_i][region] = 0.0

    case_green = [sum(s.values()) / len(s) if s else 0.0 for s in per_case]

    region_tot, region_cnt = {}, {}
    for i, row in df.iterrows():
        for region in to_dict(row["gt"]).keys():
            region_tot[region] = region_tot.get(region, 0.0) + per_case[i].get(region, 0.0)
            region_cnt[region] = region_cnt.get(region, 0) + 1

    summary = {
        "green": sum(case_green) / len(case_green) if case_green else 0.0,
        "region_means": {r: region_tot[r] / region_cnt[r] for r in sorted(region_tot)},
        "n_cases": len(df),
        "report_parse_rate": n_parsed / max(len(df), 1),
        "n_pairs_scored": len(refs),
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(summary, indent=2))

    detail = a.out_json.with_name(a.out_json.stem + "_per_case.json")
    detail.write_text(json.dumps(
        [{"case": str(df.iloc[i].get("name", i)), "green": case_green[i],
          "regions": per_case[i]} for i in range(len(df))], indent=1))

    print(f"GREEN = {summary['green']:.4f}   parse_rate = {summary['report_parse_rate']:.3f}")
    print(f"saved -> {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
