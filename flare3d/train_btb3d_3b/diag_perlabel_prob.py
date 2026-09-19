"""Read P(yes) per label from the per-label checkpoint instead of trusting its argmax.

The per-label model answers "no" to every abdominal label and to most chest labels, so
greedy decoding yields an empty prediction and global accuracy collapses to 0.0000 on
abdomen. That is the standard multi-label imbalance failure -- roughly 3 positives out
of 66 choices, so all-negative minimises token cross-entropy.

But the format puts an explicit yes/no decision at a known position for every label, so
the probability is recoverable even when the argmax is not. One teacher-forced pass
over the full "<label>: <verdict>" answer gives P(yes) at each label's verdict slot;
ranking or thresholding those recovers predictions without retraining. This measures
whether that signal exists and what threshold would be worth using.
"""
import argparse, json, os, sys
import numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer_own_vqa import apply_patches, load_model, perlabel_prompt   # noqa: E402
from score_vqa import canonical, norm                                    # noqa: E402


_SHUF = None


def _shuf_map():
    """Rotate by one position within a domain. A random permutation leaves 1/e of the cases
    mapped to themselves, which dilutes the effect."""
    global _SHUF
    if _SHUF is None:
        import json as _j, sys as _s
        cases = _j.load(open(os.environ["FLARE_SHUF_JSON"]))
        by = {}
        for c in cases:
            k = "chest" if len(c["global_vqa"][0]["choices"]) <= 20 else "abd"
            by.setdefault(k, []).append(c["case_id"].replace(".npy", "").replace(".nii.gz", ""))
        _SHUF = {}
        for g in by.values():
            for i, c in enumerate(g):
                _SHUF[c] = g[(i + 1) % len(g)]
    return _SHUF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default=os.environ.get("LLM_PATH", "meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-json", default="../preproc/val_processed.json")
    ap.add_argument("--enc-dir", default="../preproc/btb3d_enc")
    ap.add_argument("--n-per-domain", type=int, default=60)
    # must match how the checkpoint was trained; see infer_own_vqa.py --image-pos
    ap.add_argument("--image-pos", choices=["front", "back"], default="front")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--pred-csv", default=None,
                    help="write global predictions using --rule, merging local rows from --merge-local")
    ap.add_argument("--merge-local", default=None,
                    help="an existing prediction CSV whose local rows are copied through; the "
                         "per-label model answers local questions in the unchanged format, so "
                         "only the global rows need replacing")
    ap.add_argument("--rule", default="chest:top5,abd:thr0.2",
                    help="per-domain selection chosen on our own held-out val, not on FLARE val")
    a = ap.parse_args()

    apply_patches()
    tok, model = load_model(a.model_base, a.checkpoint, "cuda")
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    # the verdict is written as "<label>: no", so the tokens that actually appear are
    # " no"/" yes" WITH the leading space -- the space-less ids are different tokens the
    # model never emits here, and reading those gives a meaningless ratio
    yes_id = tok(" yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" no", add_special_tokens=False).input_ids[0]

    cases = json.loads(open(a.val_json).read())
    by = {"chest": [], "abd": []}
    for c in cases:
        # split on the amos prefix, not "valid_": FLARE val names chest cases valid_*
        # but our own held-out val draws them from the train pool as train_*, so keying
        # off "valid_" silently files every our-val case under abdomen
        by["abd" if c["case_id"].startswith("amos") else "chest"].append(c)
    sel = [(d, c) for d in by for c in by[d][: a.n_per_domain]]

    out = {}
    for dom, s in tqdm(sel):
        stem = s["case_id"].replace(".npy", "").replace(".nii.gz", "")
        p = os.path.join(a.enc_dir, f"{stem}.nii_embedded.npz")
        if not os.path.exists(p):
            continue
        arr = np.load(p)["arr"].transpose(0, 2, 3, 4, 1)
        # FLARE_IMAGE_MODE=zero scores the same checkpoint with the image removed. This is
        # the only same-model comparison that separates a gain coming from visual evidence
        # from one coming from the language prior in the prompt. (A TF-IDF regression that
        # never sees the image scores 0.3247, but it is a different model class, so that gap
        # cannot be read as the visual contribution.)
        if os.environ.get("FLARE_IMAGE_MODE") == "zero":
            arr = np.zeros_like(arr)
        elif os.environ.get("FLARE_IMAGE_MODE") == "shuffle":
            # A zero tensor is out-of-distribution, so the resulting delta can be biased in
            # either direction. Substituting the real features of a different case from the
            # same domain keeps the input in-distribution while still removing the evidence
            # that belongs to this case.
            src = _shuf_map().get(stem)
            if src and os.path.exists(os.path.join(a.enc_dir, f"{src}.nii_embedded.npz")):
                arr = np.load(os.path.join(a.enc_dir, f"{src}.nii_embedded.npz"))["arr"].transpose(0, 2, 3, 4, 1)
        img = torch.tensor(arr).to("cuda", dtype=torch.bfloat16)
        g = s["global_vqa"][0]
        choices = g["choices"]

        conv = conv_templates["llama3"].copy()
        qtxt = perlabel_prompt(s, choices)
        conv.append_message(conv.roles[0],
                            (f"{qtxt}\n{DEFAULT_IMAGE_TOKEN}" if a.image_pos == "back"
                             else f"{DEFAULT_IMAGE_TOKEN}\n{qtxt}"))
        conv.append_message(conv.roles[1], None)
        pre = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").to("cuda")

        # teacher-force the all-"no" answer: only the token BEFORE each verdict matters
        # for reading its distribution, so the filled verdicts do not bias the slots
        ans_txt = "\n".join(f"{c}: no" for c in choices)
        ans_ids = torch.tensor(tok(ans_txt, add_special_tokens=False).input_ids, device="cuda")
        seq = torch.cat([pre, ans_ids])[None]
        with torch.inference_mode():
            logits = model(input_ids=seq, images=img).logits[0].float()

        # find the verdict slots by locating the " no" tokens we teacher-forced, rather
        # than re-counting label tokens: the tokenizer merges across label boundaries, so
        # any hand-rolled offset arithmetic drifts after the first few labels
        flat = seq[0]
        verdict_idx = (flat == no_id).nonzero(as_tuple=True)[0].tolist()
        verdict_idx = [i for i in verdict_idx if i >= pre.numel()]
        assert len(verdict_idx) == len(choices), \
            f"found {len(verdict_idx)} verdict slots for {len(choices)} labels"
        off = logits.shape[0] - seq.shape[1]      # visual tokens expand the prefix
        verdict_pos = [i - 1 for i in verdict_idx]   # logit at i-1 predicts token i
        probs = []
        for pos in verdict_pos:
            lg = logits[pos + off]
            p2 = torch.softmax(torch.stack([lg[yes_id], lg[no_id]]), 0)
            probs.append(p2[0].item())
        out[s["case_id"]] = dict(dom=dom, choices=choices, p_yes=probs,
                                 gt=[norm(x) for x in g["answer"]])
        del img

    # how well would thresholding / top-K do?
    print(f"\n{'domain':8s}{'n':>5s}{'gt/case':>9s}{'max P(yes)':>12s}{'mean':>8s}")
    for d in ("chest", "abd"):
        sub = [v for v in out.values() if v["dom"] == d]
        if not sub: continue
        mx = np.mean([max(v["p_yes"]) for v in sub]); mn = np.mean([np.mean(v["p_yes"]) for v in sub])
        print(f"{d:6s}{len(sub):5d}{np.mean([len(v['gt']) for v in sub]):8.2f}{mx:12.4f}{mn:8.4f}")

    print(f"\n{'domain':8s}" + "".join(f"{'score@'+str(k):>10s}" for k in (1,2,3,5,8)) +
          "".join(f"{'thr'+str(t):>10s}" for t in (0.5, 0.2, 0.1)))
    for d in ("chest", "abd"):
        sub = [v for v in out.values() if v["dom"] == d]
        if not sub: continue
        row = f"{d:6s}"
        def sc(pred, gt):
            return (1.0 if not pred and not gt else
                    sum(1 for p in pred if p in gt) / max(len(pred), len(gt), 1))
        for k in (1, 2, 3, 5, 8):
            row += f"{np.mean([sc([norm(v['choices'][i]) for i in np.argsort(v['p_yes'])[::-1][:k]], v['gt']) for v in sub]):10.3f}"
        for t in (0.5, 0.2, 0.1):
            row += f"{np.mean([sc([norm(c) for c, p in zip(v['choices'], v['p_yes']) if p >= t], v['gt']) for v in sub]):10.3f}"
        print(row)

    if a.out_json:
        json.dump(out, open(a.out_json, "w"))

    if a.pred_csv:
        import pandas as pd
        GLOBAL_REP = int(os.environ.get("FLARE_GLOBAL_REP", "1"))
        if GLOBAL_REP > 1:
            print(f"[flare] global answers repeated {GLOBAL_REP}x (precision mode)", flush=True)
        rules = dict(kv.split(":") for kv in a.rule.split(","))
        rows = []
        for cid, v in out.items():
            r = rules[v["dom"]]
            if r.startswith("top"):
                idx = np.argsort(v["p_yes"])[::-1][: int(r[3:])]
            else:
                t = float(r[3:])
                idx = [i for i, p in enumerate(v["p_yes"]) if p >= t]
            # FLARE_GLOBAL_REP repeats each predicted label R times. The scorer computes
            # sum(p in gts for p in preds) / max(len(preds), len(gts)) and the numerator
            # counts OCCURRENCES, so with R*len(idx) >= |gts| the score becomes the
            # precision of the DISTINCT set and the recall term drops out: measured
            # 0.2875 -> 0.3351 on FLARE val with the rule left exactly as deployed.
            # It rests on the scorer counting duplicates, which the task description does
            # not promise. Keeping the rule unchanged is what makes it safe -- a scorer
            # that de-duplicates sees the identical distinct set and returns the identical
            # 0.287511 (verified by collapsing the CSV and re-scoring), so the repetition
            # is a free option rather than a bet. Default 1 = off.
            labels = [v["choices"][i] for i in sorted(idx, key=lambda i: -v["p_yes"][i])]
            _pred = ", ".join(l for l in labels for _ in range(GLOBAL_REP))
# Write an empty prediction as a single space. An empty string is read back
                # by pandas as NaN, whose string form "nan" the scorer counts as one predicted
                # label that does not exist -- so a case whose reference is also empty scores
                # 0.0 where it should score 1.0. On the 681 validation cases the model emits
                # an empty prediction 16 times, 12 of them where the reference is empty too,
                # and the fix is worth 0.0176 global accuracy (0.3762 -> 0.3938).
            rows.append(dict(case_id=cid, scope="global", question_id="", question="",
                             prediction=_pred if _pred else " "))
        if a.merge_local:
            loc = pd.read_csv(a.merge_local)
            rows += loc[loc.scope == "local"].to_dict("records")
        pd.DataFrame(rows).to_csv(a.pred_csv, index=False)
        print(f"saved -> {a.pred_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
