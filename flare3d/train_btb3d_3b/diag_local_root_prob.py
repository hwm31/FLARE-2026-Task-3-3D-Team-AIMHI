"""P(yes) at each local chain's ROOT question.

Why this exists. The local metric gates on the root: get it wrong and the chain scores 0 no
matter how good the follow-ups are. Measured on stage2_rag over 2,891 chains:

    root failed        38.3%   -> 0 points
    root passed        61.7%   -> those chains average 0.8370

so local = 0.617 x 0.837 = 0.5162, and the whole axis is a yes/no classification problem.
And on that problem the model adds nothing: the best per-chain-length constant answer scores
61.6% and the model scores 61.7%. It is also badly biased -- it says yes 73% of the time
against a 55% base rate, and the bias flips by chain length (88.8% yes on length-2 chains
against 56.7% true, 17.1% on length-3 against 36.1%).

Generation cannot fix a calibration problem: it emits one token, take it or leave it. The
same situation on chest global -- generated answers stuck at the constant-answer ceiling --
was worth +0.055 once the answer was chosen by thresholding a probability instead. This
dumps that probability so the same fix can be tried here.

Teacher-forced: the prompt is built exactly as infer_own_vqa builds it, and the yes/no
distribution is read at the position where generation would start. Nothing is generated, so
one pass covers every chain in a case.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from infer_own_vqa import apply_patches, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default=os.environ.get("LLM_PATH", "meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--image-pos", choices=["front", "back"], default="front")
    a = ap.parse_args()

    apply_patches()
    tok, model = load_model(a.model_base, a.checkpoint, "cuda")
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    # the ids the model would actually emit first. Both capitalisations are checked because
    # a wrong id here silently produces a constant probability and would look like "the
    # model has no signal" rather than "the probe is broken".
    def first_id(word):
        ids = tok.encode(word, add_special_tokens=False)
        return ids[0]
    YES, NO = first_id("Yes"), first_id("No")
    print(f"token ids: Yes={YES} No={NO}", flush=True)

    data = json.load(open(a.val_json))
    recs = data if isinstance(data, list) else [dict(case_id=k, **v) for k, v in data.items()]
    out = {}
    for s in tqdm(recs, ncols=80):
        cid = s["case_id"]
        stem = cid.replace(".npy", "").replace(".nii.gz", "")
        enc = os.path.join(a.enc_dir, f"{stem}.nii_embedded.npz")
        if not os.path.exists(enc):
            continue
        # same layout the inference path builds: (1,T,H,W,C)
        arr = np.load(enc)["arr"].transpose(0, 2, 3, 4, 1)
        img = torch.tensor(arr).to("cuda", torch.bfloat16)
        id2 = {}
        for q in s["local_vqa"]:
            id2.setdefault(q["follow_up"], []).append(q)
        per = {}
        for root in [q for q in s["local_vqa"] if q["follow_up"] == -1]:
            chain = [root] + id2.get(root["id"], [])
            lines = []
            for i, q in enumerate(chain, 1):
                line = q["question"].rstrip()
                if q.get("choices"):
                    line = f"{line} Choices: {q['choices']}"
                lines.append(f"{i}. {line}")
            q_text = "\n".join(lines)
            conv = conv_templates["llama3"].copy()
            conv.append_message(conv.roles[0],
                                f"{q_text}\n{DEFAULT_IMAGE_TOKEN}" if a.image_pos == "back"
                                else f"{DEFAULT_IMAGE_TOKEN}\n{q_text}")
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                        return_tensors="pt").unsqueeze(0).cuda()
            with torch.inference_mode():
                o = model(input_ids=ids, images=img, output_hidden_states=True)
            # The lm_head runs in bf16, whose ~8-bit mantissa drops the Yes and No logits
            # into the SAME bucket whenever they are within ~0.5% of each other: 4 of 14
            # chains in the smoke test came back at exactly 0.500, and several others shared
            # a value to six digits. A threshold cannot separate tied cases, so the two
            # logits are recomputed from the final hidden state in fp32. Only two rows of
            # the output matrix are needed, so it costs nothing.
            h = o.hidden_states[-1][0, -1].float()
            W = model.lm_head.weight
            lg = torch.stack([h @ W[YES].float(), h @ W[NO].float()])
            p = torch.softmax(lg, 0)[0].item()
            per[str(root["id"])] = dict(p_yes=p, chain_len=len(chain))
        out[cid] = per
        del img
    json.dump(out, open(a.out_json, "w"))
    n = sum(len(v) for v in out.values())
    print(f"saved -> {a.out_json}  ({len(out)} cases, {n} chains)", flush=True)


if __name__ == "__main__":
    main()
