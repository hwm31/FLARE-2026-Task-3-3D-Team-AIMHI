"""Per-label global turns WITH the case's own local questions in the prompt.

Why: the local questions and the global answer list are generated from the same report,
so the local question wording carries which findings the case has. Measured on FLARE val,
a TF-IDF logistic regression that reads ONLY the local question text and never sees the
image scores 0.3328 chest / 0.3126 abdomen global -- against 0.3486 / 0.2371 for the VLM
that sees the image and not the questions. Fusing the two reads 0.3309 overall against
0.2821, +0.0488 with a paired-bootstrap CI of [+0.0327, +0.0654] over cases.

That fusion is two models. Putting the local questions in the global prompt puts the same
signal inside the one model the rules allow, so this split exists to train that.

Only the global (gvqa) turns change; local turns are copied through byte-identical.
"""
from __future__ import annotations
import argparse, ast, json, re
from collections import defaultdict
from pathlib import Path

QUESTION = ("For each condition below, answer whether it is present in this scan. "
            "Answer with exactly one line per condition in the form '<condition>: yes' "
            "or '<condition>: no'.")
CTX_HEADER = "The radiologist also asked the following questions about this scan:"
CTX_HEADER_A = ("The radiologist also asked the following questions about this scan, "
                "and these were the answers:")


def choices_of(prompt: str):
    m = re.search(r"Choices:\s*(\[.*\])\s*$", prompt.strip(), re.S)
    return ast.literal_eval(m.group(1)) if m else None


def local_chain(rec):
    """Return (root question, root answer, follow-up questions); only the root answer is
    attached.

    Follow-up answers are deliberately left out. In a chain whose root is No, the follow-up
    answers are only conditionally meaningful, and the root verdict is the one quantity we
    can rely on at inference time -- it is also the value the scorer gates on.
    """
    qs = local_questions(rec)
    ans = str(rec["conversations"][1]["value"]).split("|")[0].strip()
    return (qs[0] if qs else ""), ans, qs[1:]


def local_questions(rec):
    """The numbered question lines out of one lvqa turn's prompt, choices dropped.

    The answer choices are dropped on purpose: they name the conclusion ('Cholecystitis')
    and would hand the model the label instead of the observation the question describes.
    """
    body = rec["conversations"][0]["value"].split("\n", 1)[-1]
    body = re.sub(r"\s*Choices:\s*\[.*\]\s*$", "", body.strip(), flags=re.S)
    return [re.sub(r"^\d+\.\s*", "", ln.strip()) for ln in body.splitlines() if ln.strip()]


def build_prompt(choices, ctx, with_answers=False):
    head = f"<image>\n{QUESTION}"
    if ctx:
        head += "\n" + (CTX_HEADER_A if with_answers else CTX_HEADER) + "\n" \
                + "\n".join(f"- {q}" for q in ctx)
    return f"{head}\nConditions: {'; '.join(choices)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data_clean")
    ap.add_argument("--answers", action="store_true",
                    help="also attach the root answer (Yes/No) -> stage2_perlabelctxa_*")
    a = ap.parse_args()
    for split in ("train", "val"):
        src = json.loads((a.data_dir / f"stage2_{split}.json").read_text())
        ctx = defaultdict(list)
        for r in src:
            if not r["id"].startswith("lvqa"):
                continue
            if a.answers:
                root, ans, kids = local_chain(r)
                if root:
                    ctx[r["image"]].append(f"{root} -> {ans}")
                ctx[r["image"]] += kids
            else:
                ctx[r["image"]] += local_questions(r)
        out, n_g, n_ctx = [], 0, 0
        for r in src:
            if not r["id"].startswith("gvqa"):
                out.append(r); continue
            choices = choices_of(r["conversations"][0]["value"])
            if choices is None:
                continue
            raw = r["conversations"][1]["value"].strip()
            present = set() if raw.lower() == "none" else {x.strip() for x in raw.split(",") if x.strip()}
            unknown = present - set(choices)
            assert not unknown, f"{r['id']}: answers absent from the choice list: {unknown}"
            c, _seen = [], set()
            for q in ctx.get(r["image"], []):
                if q not in _seen:
                    _seen.add(q); c.append(q)
            n_g += 1; n_ctx += len(c)
            out.append({
                "id": r["id"].replace("gvqa", "plcvqa", 1),
                "image": r["image"],
                "conversations": [
                    {"from": "human", "value": build_prompt(choices, c, a.answers)},
                    {"from": "gpt", "value": "\n".join(
                        f"{x}: {'yes' if x in present else 'no'}" for x in choices)},
                ],
            })
        tag = "perlabelctxa" if a.answers else "perlabelctx"
        (a.data_dir / f"stage2_{tag}_{split}.json").write_text(json.dumps(out, indent=1))
        print(f"{split}: {len(out)} turns  (global rewritten {n_g}, "
              f"avg local questions in prompt {n_ctx/max(n_g,1):.1f})")


if __name__ == "__main__":
    main()
