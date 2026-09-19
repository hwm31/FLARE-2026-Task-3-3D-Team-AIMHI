"""Merge the global and local prediction CSVs into the single file the scorer reads.

Both inputs use the five columns case_id, scope, question_id, question, prediction. The global file
contributes its `global` rows and the local file its `local` rows.

An empty prediction is written as a single space, not as an empty string. An empty CSV field is read
back by pandas as NaN, whose string form "nan" the scorer counts as one predicted label that does not
exist, so a case whose reference is also empty scores 0 instead of 1. On the 681 validation cases the
global adapter predicts nothing for 16 of them, 12 of which have an empty reference, and writing a space
is worth 0.0176 global accuracy (0.3762 -> 0.3938).
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-csv", required=True)
    ap.add_argument("--local-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()
    g = pd.read_csv(a.global_csv)
    l = pd.read_csv(a.local_csv)
    out = pd.concat([g[g.scope == "global"], l[l.scope == "local"]], ignore_index=True)
    out["prediction"] = out["prediction"].fillna(" ").replace("", " ")
    out.to_csv(a.out_csv, index=False)
    print(f"{a.out_csv}: {len(out)} rows  {out.scope.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
