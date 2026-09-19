#!/usr/bin/env python
"""VQA inference with our own trained model (BTB3D-8 encoder + Llama-3.2-3B +
Stage-1 projector + Stage-2 LoRA) on the FLARE26 3D validation set.

Loading mirrors BTB3D-8's own `infer_repgen.py::load_btb3d_lora` pattern (base model
-> add the 4 task special tokens -> resize embeddings -> load non_lora_trainables.bin
-> wrap with PeftModel -> merge_and_unload), with the same two monkeypatches
`launch_train.py` applies during training so the projector's channel count (72,
not the stale hardcoded 18) and pooling (512 queries, not the disabled flatten+MLP)
match what the checkpoint was actually trained with.

Prompts follow the exact template `train_btb3d_3b/build_instructions.py` used to
build vqa_instructions.json, so this is a faithful "ask it what it was trained to
answer" pass -- not the zero-shot region-listing prompt the baseline adapters use.
"""
import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# `llava` normally resolves through `pip install -e BTB3D/report-generation`. Setting BTB3D_DIR
# additionally puts that checkout first on sys.path.
BTB3D_REPGEN = os.path.join(os.environ["BTB3D_DIR"], "report-generation") if os.environ.get("BTB3D_DIR") else None
if BTB3D_REPGEN:
    sys.path.insert(0, BTB3D_REPGEN)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FLARE_MM_HIDDEN = int(os.environ.get("FLARE_MM_HIDDEN", 72))
FLARE_N_QUERIES = int(os.environ.get("FLARE_N_QUERIES", 512))


def apply_patches():
    """Same two monkeypatches as launch_train.py -- see that file's docstring."""
    import torch as _torch
    from llava.model import llava_arch
    from llava.model.multimodal_projector.builder import build_vision_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        self.config.mm_vision_tower = model_args.vision_tower
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = FLARE_MM_HIDDEN
        self.config.mm_context_size = FLARE_MM_HIDDEN
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.config.mm_patch_merge_type = model_args.mm_patch_merge_type
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True
    llava_arch.LlavaMetaModel.initialize_vision_modules = initialize_vision_modules

    from spatial_projector import install_projector_patch
    install_projector_patch(FLARE_N_QUERIES)


def load_model(model_base, checkpoint_dir, device):
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.train.train import smart_tokenizer_and_embedding_resize
    from llava.constants import (TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER,
                                 TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION)
    from peft import PeftModel

    cfg = LlavaConfig.from_pretrained(model_base)
    cfg.mm_vision_tower = "openai/clip-vit-large-patch14-336"
    cfg.mm_projector_type = "attn_pool+mlp2x_gelu"
    cfg.mm_hidden_size = FLARE_MM_HIDDEN
    cfg.mm_context_size = FLARE_MM_HIDDEN
    cfg.mm_vision_select_layer = -2
    cfg.mm_patch_merge_type = "flat"
    cfg.mm_use_im_start_end = False
    cfg.pad_token_id = None

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_base, low_cpu_mem_usage=True, config=cfg,
        torch_dtype=torch.bfloat16, device_map={"": device})

    from llava.model.multimodal_projector.builder import build_vision_projector
    # nn.Module params default to float32; the rest of the model was loaded in bf16,
    # so cast the freshly-built projector before loading checkpoint weights into it,
    # or load_state_dict's copy_() silently keeps it in float32 -> dtype crash later.
    model.get_model().mm_projector = build_vision_projector(model.config).to(
        device=device, dtype=torch.bfloat16)

    smart_tokenizer_and_embedding_resize(dict(pad_token="<pad>"), tokenizer, model)
    for tok in (TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER,
                TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION):
        tokenizer.add_tokens(tok, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    nlt = torch.load(os.path.join(checkpoint_dir, "non_lora_trainables.bin"), map_location="cpu")
    nlt = {(k[len("base_model.model."):] if k.startswith("base_model.model.") else k): v
           for k, v in nlt.items()}
    missing, unexpected = model.load_state_dict(nlt, strict=False)
    print(f"[load] non_lora_trainables: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # An "unexpected" key means the checkpoint trained a module this process did not build
    # -- e.g. loading an aux-head or injection run without FLARE_AUX_CLS / FLARE_INJECT set.
    # load_state_dict drops those weights and returns success, so the run would score a
    # different model than the one that was trained. Fail instead.
    assert not unexpected, (
        f"checkpoint has {len(unexpected)} tensors this model lacks, e.g. {unexpected[:3]}. "
        "Set the same FLARE_* env vars used for training so the modules are built.")

    # The mirror check, and the one that was missing. `unexpected` catches a checkpoint
    # that trained MORE than we built; it does not catch a checkpoint that trained a
    # DIFFERENT projector whose submodules happen to share names. A `direct` run stores
    # mm_projector.ln.* and mm_projector.proj.*, and an AttentionalPooler has an `ln` and a
    # `proj` too -- so loading one into the other reports unexpected=0 while leaving every
    # attn_pool.* tensor randomly initialised. That scored proj32_direct_s7 at global
    # 0.1349 against its true 0.2830, and nothing in the output said anything was wrong.
    #
    # So: every mm_projector parameter the model has must actually come from the file.
    built = {k for k, _ in model.get_model().mm_projector.named_parameters()}
    loaded = {k.split("mm_projector.", 1)[1] for k in nlt if "mm_projector." in k}
    absent = built - loaded
    assert not absent, (
        f"{len(absent)}/{len(built)} projector tensors are NOT in the checkpoint, e.g. "
        f"{sorted(absent)[:3]} -- this checkpoint was trained with a different projector. "
        f"Set FLARE_PROJ to the one it was trained with (currently "
        f"{os.environ.get('FLARE_PROJ') or 'unset = AttentionalPooler'}).")

    model = PeftModel.from_pretrained(model, checkpoint_dir, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    # Injection hook, installed on the BUILT model rather than by wrapping
    # LlavaLlamaForCausalLM.forward. Wrapping forward with (*args, **kwargs) erases its
    # signature, and transformers' generate introspects that signature to decide which
    # kwargs to thread through -- the mask then never reaches _update_model_kwargs_for_
    # generation and it dies on `attention_mask.new_ones` with attention_mask None.
    #
    # Without this the trained inject_head still LOADS, is never called, and scoring
    # silently reports the injection's result for a model with the injection off.
    if os.environ.get("FLARE_INJECT"):
        proj = model.get_model().mm_projector
        if getattr(proj, "inject_head", None) is not None:
            def _pre_hook(_m, args):
                h = args[0]
                f = getattr(proj, "_aux_feat", None)
                if f is None or f.shape[0] != h.shape[0]:
                    return None
                add = proj.inject_head(f.to(h.dtype)) * proj.inject_gate.to(h.dtype)
                return (h + add[:, None, :],)
            model.lm_head.register_forward_pre_hook(_pre_hook)
            print("[flare-patch] injection hook active at inference", flush=True)
        else:
            raise SystemExit("FLARE_INJECT set but the checkpoint has no inject_head")
    return tokenizer, model.to(device).eval()


PERLABEL_QUESTION = (
    "For each condition below, answer whether it is present in this scan. "
    "Answer with exactly one line per condition in the form '<condition>: yes' "
    "or '<condition>: no'.")

CTX_HEADER = "The radiologist also asked the following questions about this scan:"
CTX_HEADER_A = ("The radiologist also asked the following questions about this scan, "
                "and these were the answers:")
_ROOTANS = None


def _root_answers():
    """Root answers to insert into the global prompt at inference. The JSON that
    FLARE_ROOT_ANS points at is exactly the local output we submit -- the first stage of the
    same pipeline, not a separate model -- so this is two-pass inference, not an ensemble."""
    global _ROOTANS
    if _ROOTANS is None:
        p = os.environ.get("FLARE_ROOT_ANS", "")
        _ROOTANS = json.load(open(p)) if p and os.path.exists(p) else {}
    return _ROOTANS


def _stem(cid):
    return str(cid).replace(".nii.gz", "").replace(".npy", "")


_QSHUF = None


def _shuffled_questions(sample):
    """Serve another case's local questions, to separate whether the prompt-fusion gain comes
    from the case-specific content of the questions or merely from their format and length.
    Shuffling stays within a domain: chest and abdomen have different label lists, so the
    model can tell them apart without the image, and crossing domains would measure something
    else."""
    global _QSHUF
    if _QSHUF is None:
        import json as _j
        cases = _j.load(open(os.environ["FLARE_QSHUF_SRC"]))
        by = {}
        for c in cases:
            k = "chest" if len(c["global_vqa"][0]["choices"]) <= 20 else "abd"
            by.setdefault(k, []).append(c)
        _QSHUF = {}
        for k, g in by.items():
            for i, c in enumerate(g):          # rotate: no case receives its own questions
                _QSHUF[_stem(c["case_id"])] = g[(i + 1) % len(g)]["local_vqa"]
    return _QSHUF.get(_stem(sample.get("case_id", "")), sample.get("local_vqa", []))


def perlabel_prompt(sample, choices):
    """The global per-label prompt, optionally carrying the case's own local questions.

    Set FLARE_GLOBAL_CTX=1 to include them -- it must match how the checkpoint was
    trained (train_btb3d_3b/build_perlabelctx_split.py builds the matching split), so it
    is an env flag read in one place and used by every scoring path, not a per-script
    string that can drift.
    """
    q = PERLABEL_QUESTION
    mode = os.environ.get("FLARE_GLOBAL_CTX", "0")
    if mode in ("1", "2"):
        # grouped root-then-follow-up, which is the order the training split emits
        # (it reads one lvqa turn at a time, and a turn holds exactly one chain)
        lv = (_shuffled_questions(sample) if os.environ.get("FLARE_QSHUF_SRC")
              else sample.get("local_vqa", []))
        seen, ctx = set(), []
        ra = _root_answers().get(_stem(sample.get("case_id", "")), {}) if mode == "2" else {}
        # FLARE_CTX_TRUNC=1 reproduces the list the training split actually contains.
        # The regex in build_perlabelctx_split.local_questions,
        #   re.sub(r"\s*Choices:\s*\[.*\]\s*$", "", body, flags=re.S)
        # is greedy, so it deleted everything from the first 'Choices:' onwards and dropped
        # the third question from 98.1% of three-question chains. Inference includes them all,
        # so the training and inference prompts differ. This switch exists to measure which is
        # better; the measured difference is 0.3935 against 0.3938, i.e. none.
        trunc = os.environ.get("FLARE_CTX_TRUNC", "0") == "1"
        for root in [x for x in lv if x["follow_up"] == -1]:
            kids = [k for k in lv if k["follow_up"] == root["id"]]
            if trunc:
                kids = kids[:1]
            for item in [root] + kids:
                t = str(item["question"]).strip()
                if t in seen:
                    continue
                seen.add(t)
                a = ra.get(str(root["id"])) if item is root else None
                ctx.append(f"{t} -> {a}" if a else t)
        if ctx:
            head = CTX_HEADER_A if mode == "2" else CTX_HEADER
            q += "\n" + head + "\n" + "\n".join(f"- {t}" for t in ctx)
    return f"{q}\nConditions: {'; '.join(choices)}"


def parse_perlabel(text, choices):
    """'<condition>: yes' lines -> the comma list the scorer parses.

    Returns "" and not "None" when nothing is present: score_vqa treats an empty
    prediction against an empty ground truth as a full 1.0, but reads the literal
    "None" as one predicted finding and scores 0. 35 of the 681 val cases have an
    empty ground truth, so that distinction is worth ~0.05 global on its own.
    """
    lower = {c.lower(): c for c in choices}
    present = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, verdict = line.rpartition(":")
        key = name.strip().lower()
        if verdict.strip().lower().startswith("yes") and key in lower:
            present.append(lower[key])
    return ", ".join(present)


def build_shuffle_map(cases, seed):
    """Map each case to a DIFFERENT case's encoding, staying within the same lineage.

    Within-lineage on purpose: the two lineages get different `choices` lists in the
    prompt, so the model can tell chest from abdomen without looking at the image at
    all. Handing an abdominal case a chest scan would therefore measure a confound.
    Swapping patients inside one lineage keeps the input distribution and the prompt
    identical and changes only which patient the scan belongs to -- so if predictions
    do not move, the model is not reading the image.

    A derangement (no case keeps its own scan) via a rotation by one, since a random
    permutation leaves ~1/e of cases mapped to themselves, which would dilute the
    effect being measured.
    """
    import random
    groups = {}
    for c in cases:
        cid = c["case_id"]
        groups.setdefault("chest" if "valid_" in cid else "abd", []).append(cid)
    mapping = {}
    for lineage, ids in groups.items():
        ids = sorted(ids)
        random.Random(seed).shuffle(ids)
        for i, cid in enumerate(ids):
            mapping[cid] = ids[(i + 1) % len(ids)]
    assert all(k != v for k, v in mapping.items()), "shuffle left a case with its own scan"
    return mapping


def strip_findings(pred: str) -> str:
    """Keep only what follows the 'Answer:' marker of a findings-first generation.

    The marker is explicit in the target rather than inferred, so scoring never has to
    guess where the findings list ends and the chain begins. If the model omits it -- which
    a partly-trained checkpoint will sometimes do -- the raw text is returned so the failure
    shows up as a wrong answer in the score rather than as a silently emptied prediction.
    """
    for line in pred.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return pred


def format_pred(pred: str) -> str:
    lines = [l.strip() for l in pred.splitlines() if l.strip()]
    numbered = [re.sub(r"^\s*\d+\.\s*", "", l).strip()
                for l in lines if re.match(r"^\s*\d+\.\s*", l)]
    return "|".join(numbered) if numbered and len(numbered) == len(lines) else pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base",
                    default=os.environ.get("LLM_PATH", "meta-llama/Llama-3.2-3B-Instruct"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--proj-tokens", type=int, default=FLARE_N_QUERIES)
    ap.add_argument("--do-sample", action="store_true",
                    help="sample instead of greedy; used to test whether the single-answer "
                         "collapse is a decoding artifact or baked into the weights")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overlay", default=None,
                    help="a GRPO trainable.pt whose tensors are applied on top of the "
                         "checkpoint; RL updates only a subset of the weights, so the "
                         "base checkpoint still has to be loaded first")
    ap.add_argument("--oracle-findings", action="store_true",
                    help="prefill the assistant turn with the TRUE findings, then generate "
                         "only the answer. An upper bound on findings-first, not a "
                         "submittable configuration -- it reads the labels it is scored on.")
    ap.add_argument("--local-format", choices=["plain", "findings"], default="plain",
                    help="findings = the model writes 'Findings: ...' then 'Answer: ...'; "
                         "scoring keeps only the part after the marker")
    ap.add_argument("--batch-local", type=int, default=1,
                    help="how many chains of one case to generate together. Default 1\n"
                         "(sequential). Batching is about 3x faster, but with left-padded\n"
                         "batches LLaVA's multimodal input reconstruction goes wrong and 12%%\n"
                         "of outputs change (unchanged after two attempts to fix the decode\n"
                         "step). Sequential generation stays inside the challenge runtime\n"
                         "limit, so batching is left off.")
    ap.add_argument("--image-pos", choices=["front", "back"], default="front",
                    help="where the visual block sits in the prompt; must match training")
    ap.add_argument("--image-mode", choices=["real", "shuffle", "zero"], default="real",
                    help="ablation: does the model actually read the image? 'shuffle' feeds "
                         "another case's encoding, 'zero' feeds zeros. See build_shuffle_map().")
    ap.add_argument("--shuffle-seed", type=int, default=1234)
    ap.add_argument("--global-format", choices=["freeform", "perlabel"], default="freeform",
                    help="perlabel matches build_perlabel_split.py: ask for a yes/no verdict "
                         "on every choice, then parse the 'yes' lines back into the comma "
                         "list the scorer expects")
    a = ap.parse_args()

    apply_patches()
    device = "cuda"
    tokenizer, model = load_model(a.model_base, a.checkpoint, device)
    if a.overlay:
        ov = torch.load(a.overlay, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            {k: v.to(dtype=torch.bfloat16) for k, v in ov.items()}, strict=False)
        assert not unexpected, f"overlay has keys the model lacks: {unexpected[:3]}"
        print(f"[overlay] applied {len(ov)} tensors from {a.overlay}", flush=True)

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    # ONE literal "<image>", not a run of <im_patch>. tokenizer_image_token splits the
    # prompt on the string "<image>" and puts a single IMAGE_TOKEN_INDEX at each split;
    # prepare_inputs_labels_for_multimodal then expands that one slot into all 512
    # pooled visual tokens. A prompt without the literal "<image>" yields zero slots,
    # and the num_images==0 branch of llava_arch splices cur_image_features[0:0] --
    # an empty tensor -- so the model runs blind on text alone. That was the bug: every
    # inference run before 2026-08-19 fed 512 copies of the *string* "<im_patch>" and no
    # image at all, which is why greedy global answers were constant per lineage.
    def ask(image_tensor, question, max_new_tokens, prefill=None):
        """prefill starts the assistant turn with text the model does not have to produce.

        Used for the oracle-findings condition: a findings-first checkpoint is trained to
        write "Findings: ...\nAnswer: ...", so writing the TRUE findings into that slot and
        letting it generate only the answer asks a question no other run can -- given
        perfect visual information, in the exact format this model was trained to consume,
        does the answer get better? If it does not, the ceiling is not the model's ability
        to see, and the whole findings-conditioning direction is closed.
        """
        conv = conv_templates["llama3"].copy()
        # --image-pos must match how the checkpoint was TRAINED: a model trained with the
        # visual block at the end of the prompt reads it from an adjacent position, and
        # asking it front-first is a different input distribution, not a fair score.
        conv.append_message(conv.roles[0],
                            f"{question}\n{DEFAULT_IMAGE_TOKEN}" if a.image_pos == "back"
                            else f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if prefill:
            prompt = prompt + prefill
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)
        # Guard the bug above: no slot means the image is silently dropped and the run
        # still "succeeds", producing plausible text-only output. Fail loudly instead.
        n_slots = int((input_ids == IMAGE_TOKEN_INDEX).sum())
        assert n_slots == 1, f"expected exactly 1 image slot in the prompt, got {n_slots}"
        gen_kw = dict(max_new_tokens=max_new_tokens, use_cache=True)
        if a.do_sample:
            gen_kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p)
        else:
            gen_kw.update(do_sample=False)
        with torch.inference_mode():
            out = model.generate(input_ids, images=image_tensor, **gen_kw)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text.replace(prompt.replace("<image>", "").strip(), "").strip()

    def ask_batch(image_tensor, questions, max_new_tokens):
        """Generate several chains of one case together.

        Motivation: a case carries 4.6 chains on average, and generating them one at a time
        is the dominant term in per-case runtime. Batching would obtain the same outputs far
        more cheaply -- but see --batch-local, which is off by default because the outputs
        are not in fact identical.

        Left padding is required: generation continues from the end of the sequence, so with
        right padding the model would continue after the padding tokens and the output would
        be corrupt. The image tensor is replicated to the batch size.
        """
        if len(questions) == 1:
            return [ask(image_tensor, questions[0], max_new_tokens)]
        prompts = []
        for q in questions:
            conv = conv_templates["llama3"].copy()
            conv.append_message(conv.roles[0],
                                f"{q}\n{DEFAULT_IMAGE_TOKEN}" if a.image_pos == "back"
                                else f"{DEFAULT_IMAGE_TOKEN}\n{q}")
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
        ids = [tokenizer_image_token(pr, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
               for pr in prompts]
        n = max(len(x) for x in ids)
        pad = tokenizer.pad_token_id or 0
        inp = torch.full((len(ids), n), pad, dtype=ids[0].dtype)
        att = torch.zeros((len(ids), n), dtype=torch.long)
        for i, x in enumerate(ids):                    # left padding
            inp[i, n - len(x):] = x
            att[i, n - len(x):] = 1
        inp, att = inp.to(device), att.to(device)
        imgs = image_tensor.expand(len(ids), *image_tensor.shape[1:])
        gen_kw = dict(max_new_tokens=max_new_tokens, use_cache=True, attention_mask=att)
        gen_kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p) if a.do_sample \
            else gen_kw.update(do_sample=False)
        with torch.inference_mode():
            out = model.generate(inp, images=imgs, **gen_kw)
        # Slicing the prompt off as a string interacts badly with left padding and corrupts
        # the result ("None of the above" truncated to "None of the", among others -- 12% of
        # outputs differed). With left padding every sequence has the same input length n, so
        # the generated part is exactly out[i][n:].
        # LLaVA's generate returns either (input + generated) or (generated only) depending on
        # the code path; decide by length rather than by string matching.
        gen = out[:, n:] if out.shape[1] > n else out
        return [tokenizer.decode(gen[i], skip_special_tokens=True).strip()
                for i in range(len(prompts))]

    cases = json.loads(open(a.val_json).read())

    def _stem(x):
        """case ids arrive as amos_0008.npy here and amos_0008.nii.gz in the label files"""
        return str(x).replace(".nii.gz", "").replace(".npy", "")

    oracle = {}
    if a.oracle_findings:
        import glob
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("flareval_chest.json", "flareval_abd.json"):
            for r in json.loads(open(os.path.join(here, "data_clean", f)).read()):
                v = r["conversations"][1]["value"].strip()
                oracle[_stem(r["image"])] = "none" if v.lower() == "none" else v
        print(f"[oracle] true findings loaded for {len(oracle)} cases", flush=True)
    if a.limit:
        cases = cases[: a.limit]

    shuffle_map = build_shuffle_map(cases, a.shuffle_seed) if a.image_mode == "shuffle" else {}

    rows = []
    for sample in tqdm(cases):
        case_id = sample["case_id"]
        src_id = shuffle_map.get(case_id, case_id)
        stem = src_id.replace(".npy", "").replace(".nii.gz", "")
        enc_path = os.path.join(a.enc_dir, f"{stem}.nii_embedded.npz")
        if not os.path.exists(enc_path):
            print(f"[warn] missing encoding for {stem}", flush=True)
            continue
        arr = np.load(enc_path)["arr"].transpose(0, 2, 3, 4, 1)  # (1,T,H,W,C)
        if a.image_mode == "zero":
            arr = np.zeros_like(arr)
        image_tensor = torch.tensor(arr).to(device, dtype=torch.bfloat16)

        for g in sample["global_vqa"]:
            if a.global_format == "perlabel" and g.get("choices"):
                q = perlabel_prompt(sample, g["choices"])
                # one line per condition: 66 abdominal labels need far more than 128
                raw = ask(image_tensor, q, 24 + 12 * len(g["choices"]))
                pred = parse_perlabel(raw, g["choices"])
            else:
                q = g["question"].rstrip()
                if g.get("choices"):
                    q = f"{q} Choices: {g['choices']}"
                pred = ask(image_tensor, q, 128)
            rows.append(dict(case_id=case_id, scope="global", question_id="",
                             question=g["question"],
                             prediction=pred if a.global_format == "perlabel" else (pred or "None")))

        locals_ = sample["local_vqa"]
        id2qs = {}
        for q in locals_:
            id2qs.setdefault(q["follow_up"], []).append(q)
        _pending = []
        for root in [q for q in locals_ if q["follow_up"] == -1]:
            chain = [root] + id2qs.get(root["id"], [])
            lines = []
            for idx, q in enumerate(chain, 1):
                line = q["question"].rstrip()
                if q.get("choices"):
                    line = f"{line} Choices: {q['choices']}"
                lines.append(f"{idx}. {line}")
            q_text = "\n".join(lines)
            if os.environ.get("FLARE_LOCAL_CTXG") == "1":
                # Same header and same position as the training prompt built by
                # build_local_ctxg_split.py. The candidate labels are part of the input, not a
                # prediction.
                gch = (sample.get("global_vqa") or [{}])[0].get("choices") or []
                if gch:
                    q_text = ("Conditions that may be present in this scan: "
                              + "; ".join(gch) + "\n" + q_text)
            if a.local_format == "findings":
                # must match build_findings_first.py's PREFIX_INSTR, and the budget has to
                # cover the findings list as well as the chain or the answer is cut off
                q_text = (q_text + "\nFirst list the abnormal findings you can see, as "
                          "'Findings: a, b, c' (write 'Findings: none' if there are none). "
                          "Then answer on the next line as 'Answer: ...'.")
            pre = None
            if a.oracle_findings:
                key = _stem(case_id)
                assert key in oracle, f"no ground-truth findings for {key}"
                pre = f"Findings: {oracle[key]}\nAnswer:"
            _pending.append((root, q_text, pre))
        # Batch the chains of one case. Chains carrying a prefill (oracle findings) have
        # different prompts and cannot be batched, so they take the sequential path.
        _budget = 160 if a.local_format == "findings" else 64
        if a.batch_local > 1 and not any(pr for _, _, pr in _pending):
            _preds = []
            for i in range(0, len(_pending), a.batch_local):
                chunk = _pending[i:i + a.batch_local]
                _preds += ask_batch(image_tensor, [q for _, q, _ in chunk], _budget)
        else:
            _preds = [ask(image_tensor, q, _budget, prefill=pr) for _, q, pr in _pending]
        for (root, _q, pre), pred in zip(_pending, _preds):
            if a.local_format == "findings" and not pre:
                pred = strip_findings(pred)
            rows.append(dict(case_id=case_id, scope="local", question_id=root["id"],
                             question=q_text, prediction=format_pred(pred) or "None"))
        del image_tensor
        pd.DataFrame(rows).to_csv(a.pred_csv, index=False)

    print(f"saved -> {a.pred_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
