#!/usr/bin/env python
"""Training entrypoint for BTB3D-encoder + Llama-3.2-3B on FLARE26 3D.

Reuses the shared BTB3D `report-generation` LLaVA training stack (LoRA config, the
lazy dataset/collator, DeepSpeed wiring) completely unmodified -- we only monkeypatch
two things IN THIS PROCESS, never on disk, so the shared checkout stays untouched for
other people's pipelines:

  1. `LlavaMetaModel.initialize_vision_modules` hardcodes
     `self.config.mm_hidden_size = 18` / `mm_context_size = 18` INLINE, then in the
     same call immediately builds the projector AND (when --pretrain_mm_mlp_adapter is
     given, i.e. our stage-2 LoRA run) loads a checkpoint into it -- all before control
     ever returns to a caller. A "call orig() then fix config after" patch is too late:
     by the time orig() returns, it has already built an 18-channel projector and
     crashed trying to load our 72-channel stage-1 checkpoint into it. So this is a
     full replacement of the method (copied from llava_arch.py, hardcoded 18 swapped
     for the true 8_8_8-folded channel count), not a wrap-and-patch.

  2. `build_vision_projector`'s attn_pool branch instantiates `AttentionalPoolProjector`
     from `coca_attentional_pooler.py`, whose forward() has the actual pooling call
     commented out ("remove attentionalpool for now") -- it just flattens+MLPs every
     voxel-code token (31,744 for 8_8_8). We swap in the working `AttentionalPooler`
     (already implemented upstream, just unused) via `attn_pool_projector.py`, with
     n_queries=512, so 31,744 tokens collapse to 512 before the LLM ever sees them.

  3. `make_supervised_data_module` hardcodes `eval_dataset=None`, so HF's
     `--evaluation_strategy steps` silently evaluates nothing -- which is exactly how
     the first Stage-2 run ended up with no held-out signal at all. We wrap it to
     build a second LazySupervisedDataset from `--eval_data_path` when that flag is
     given, so eval_loss is real and `--load_best_model_at_end` has something to rank.

Everything else -- argument parsing, `train()`, LoRA/DeepSpeed setup -- is imported
directly from `llava.train.train` and driven by the same CLI flags the stock
pretrain.sh / finetune_lora.sh scripts use.
"""
import os
import sys

# `llava` normally resolves through `pip install -e BTB3D/report-generation`. Setting BTB3D_DIR
# additionally puts that checkout first on sys.path.
BTB3D_REPGEN = os.path.join(os.environ["BTB3D_DIR"], "report-generation") if os.environ.get("BTB3D_DIR") else None
if BTB3D_REPGEN:
    sys.path.insert(0, BTB3D_REPGEN)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FLARE_MM_HIDDEN = int(os.environ.get("FLARE_MM_HIDDEN", 72))
FLARE_N_QUERIES = int(os.environ.get("FLARE_N_QUERIES", 512))


def patch_vision_modules():
    import torch
    from llava.model import llava_arch
    from llava.model.multimodal_projector.builder import build_vision_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        """Faithful copy of LlavaMetaModel.initialize_vision_modules, with the
        hardcoded 18/18 channel count replaced by FLARE_MM_HIDDEN. See module
        docstring for why this must be a full replacement, not a wrap."""
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
            if "unpad" in model_args.mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = torch.nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter is not None:
            weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
            def get_w(w, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in w.items() if keyword in k}
            missing, unexpected = self.mm_projector.load_state_dict(
                get_w(weights, "mm_projector"), strict=False)
            print(f"[flare-patch] loaded pretrain_mm_mlp_adapter: "
                 f"missing={missing} unexpected={unexpected}", flush=True)

    llava_arch.LlavaMetaModel.initialize_vision_modules = initialize_vision_modules
    print(f"[flare-patch] initialize_vision_modules replaced: "
         f"mm_hidden_size=mm_context_size={FLARE_MM_HIDDEN}", flush=True)


def patch_projector():
    """Delegates to spatial_projector.install_projector_patch so training and inference
    build the SAME module. FLARE_PROJ unset keeps the AttentionalPooler."""
    from spatial_projector import install_projector_patch
    install_projector_patch(FLARE_N_QUERIES)


def patch_eval_dataset():
    """Wire --eval_data_path into the trainer; upstream hardcodes eval_dataset=None."""
    import llava.train.train as T

    eval_path = None
    argv = sys.argv
    if "--eval_data_path" in argv:
        i = argv.index("--eval_data_path")
        eval_path = argv[i + 1]
        del argv[i:i + 2]          # DataArguments has no such field; strip before parsing
    if not eval_path:
        return

    orig = T.make_supervised_data_module

    def wrapped(tokenizer, data_args):
        mod = orig(tokenizer=tokenizer, data_args=data_args)
        import copy
        eval_args = copy.copy(data_args)
        eval_args.data_path = eval_path
        mod["eval_dataset"] = T.LazySupervisedDataset(
            tokenizer=tokenizer, data_path=eval_path, data_args=eval_args)
        print(f"[flare-patch] eval_dataset wired: {eval_path} "
             f"({len(mod['eval_dataset'])} turns)", flush=True)
        return mod

    T.make_supervised_data_module = wrapped


def patch_image_position():
    """FLARE_IMAGE_POS=back: put the visual block right before the answer, not at the front.

    Measured on stage2_perlabel (diag_signal_path.py): the last prompt token -- the one
    that generates the answer -- spends 2.8% of its attention on the 512 image positions
    that occupy 72.5% of the sequence, and its final hidden state is 99.7% cosine-identical
    across patients, even though a probe on the image positions themselves still reads
    AUC 0.595. The information survives the pipe; the answer position never collects it.

    Stock `preprocess_multimodal` (llava/train/train.py:354-357) strips the image token
    wherever the data put it and re-prepends it, so the block is always ~700 tokens away
    from where it is read. This patch keeps it wherever the data put it, which lets the
    image-last split actually train image-last. Token count and content are identical --
    only the distance to the answer changes -- so a difference between the two runs is
    attributable to position alone.
    """
    import llava.train.train as T
    from llava.constants import DEFAULT_IMAGE_TOKEN

    def preprocess_multimodal(sources, data_args):
        is_multimodal = data_args.is_multimodal
        if not is_multimodal:
            return sources
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    # exactly one slot, left where the builder placed it
                    assert sentence["value"].count(DEFAULT_IMAGE_TOKEN) == 1
        return sources

    T.preprocess_multimodal = preprocess_multimodal
    print("[flare-patch] image position: kept as authored (image-last enabled)", flush=True)


def patch_lmh_loss():
    """FLARE_LMH_LAMBDA: stop the language prior from absorbing the gradient.

    Measured on stage2_perlabel (diag_image_necessity.py, 377 held-out turns): per-token
    NLL of the gold answer is 0.0901 with the case's own scan, 0.0900 with another
    patient's, and 0.0907 with the visual tokens zeroed. Stage 2 reaches eval_loss 0.0557
    without ever needing to look, so there is no gradient left to build a read with.

    Product-of-experts debiasing (Clark et al. 2019; RUBi, Learned-Mixin): train an
    ensemble of the model and a deliberately image-blind branch, so what the blind branch
    already predicts correctly contributes almost no gradient.

        loss = CE( log_softmax(logits_model) + lam * log_bias , y )

    THE BIAS BRANCH MUST BE A DIFFERENT FUNCTION. The first version of this patch used the
    same network with the visual tokens zeroed, reasoning that it was a free question-only
    branch. It is not: because the model already ignores the image, its blind output equals
    its real output to within 0.0006 nats, so the ensemble collapsed to (1+lam) copies of
    one distribution -- a temperature change. The run did exactly what that predicts: the
    model flattened its logits (entropy 0.090 -> 1.088 at lam=1.0) and image dependence got
    WORSE, with real-beats-shuffle falling 53.8% -> 50.1%.

    So the bias is now an empirical count, P(yes | condition, lineage), from
    data_clean/label_prior.json. A table cannot converge to the model, which is the point.
    It is applied only at the yes/no answer positions; everywhere else the bias is a
    constant vector, which cancels in the softmax and so is a no-op by construction.

    Lineage is read off the number of yes/no positions in the sample (18 chest, 66 abdomen
    -- verified exact for all 2911 training answers, whose label order also matches the
    prompt exactly), so the i-th verdict position is the i-th condition. Any sample that
    does not match either count is left unbiased rather than guessed at.

    Training only: our test distribution equals train, unlike VQA-CP, so the prior is real
    information and is never subtracted at inference -- a constant answer already scores
    0.2992 on chest. Eval keeps the plain loss so it stays comparable across runs.
    """
    import json as _json
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    lam = float(os.environ["FLARE_LMH_LAMBDA"])
    prior_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data_clean/label_prior.json")
    prior = _json.load(open(prior_path))
    YES, NO = 10035, 912                       # " yes" / " no" for this tokenizer
    by_len = {len(v["labels"]): v["p_yes"] for v in prior.values()}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        if not model.training:
            return (out.loss, out) if return_outputs else out.loss
        base = model.module if hasattr(model, "module") else model
        labels = getattr(base, "_flare_expanded_labels", None)
        if labels is None:
            return (out.loss, out) if return_outputs else out.loss

        tgt = labels[:, 1:]
        sel = tgt != -100
        if not sel.any():
            return (out.loss, out) if return_outputs else out.loss
        logits = out.logits[:, :-1]
        lp_m = torch.log_softmax(logits[sel].float(), -1)

        # bias logits: zero everywhere (a no-op after renormalisation) except the two
        # verdict tokens at the positions that carry a verdict
        bias = torch.zeros_like(lp_m)
        row = 0
        for b in range(tgt.shape[0]):
            t = tgt[b][sel[b]]
            pos = ((t == YES) | (t == NO)).nonzero(as_tuple=True)[0]
            p = by_len.get(int(pos.numel()))
            if p is not None:
                pv = torch.tensor(p, device=lp_m.device, dtype=lp_m.dtype)
                bias[row + pos, YES] = pv.log()
                bias[row + pos, NO] = (1 - pv).log()
            row += int(sel[b].sum())

        ens = lp_m + lam * bias                # cross_entropy renormalises the product
        loss = F.cross_entropy(ens, tgt[sel])
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] LMH product-of-experts loss ON, lambda={lam} "
          f"(bias = empirical label prior, {sorted(by_len)} labels/lineage)", flush=True)


def patch_capture_labels():
    """Stash the post-splice labels so the LMH loss can align with the expanded logits.

    prepare_inputs_labels_for_multimodal expands the single image slot into 512 visual
    positions and pads the labels to match, but returns them only into super().forward().
    Recomputing that expansion just to recover them would run the projector a third time.
    """
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    orig = LlavaLlamaForCausalLM.forward

    def forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if kwargs.get("inputs_embeds") is None and kwargs.get("labels") is not None:
            (ids2, pos, att, pkv, emb, lab) = self.prepare_inputs_labels_for_multimodal(
                ids, kwargs.get("position_ids"), kwargs.get("attention_mask"),
                kwargs.get("past_key_values"), kwargs.get("labels"),
                kwargs.get("images"), kwargs.get("image_sizes"))
            self._flare_expanded_labels = lab
            kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("input_ids", "position_ids", "attention_mask",
                                   "past_key_values", "labels", "images", "image_sizes",
                                   "inputs_embeds")}
            return super(LlavaLlamaForCausalLM, self).forward(
                input_ids=None, position_ids=pos, attention_mask=att, past_key_values=pkv,
                inputs_embeds=emb, labels=lab, **kwargs)
        return orig(self, *args, **kwargs)

    LlavaLlamaForCausalLM.forward = forward
    print("[flare-patch] expanded labels captured for the LMH loss", flush=True)


def patch_aux_cls_loss():
    """FLARE_AUX_CLS=weight: supervise the projector directly, before the bottleneck.

    Why this and not another objective on the answer. Measured with diag_grad_flow.py on
    stage2_perlabel: the LLM loss delivers gradient RMS 2.224e-09 per element to the visual
    features and 2.770e-04 to the text embeddings -- 124,580x -- because everything flowing
    back to the projector is gated by the attention the answer position pays to the image,
    and that is 0.028 against a 0.725 uniform share. Contrastive decoding, mDPO (600 / 3000
    steps, and with the image term weighted 5x) and product-of-experts debiasing all stalled
    on that same factor; mDPO ended at 97.65% of local predictions unchanged under a
    shuffled scan, worse than the 84.7% it started from.

    A label head on the pooled visual tokens is upstream of the bottleneck, so its gradient
    reaches the projector without being divided by the attention it is trying to increase.

    The signal to supervise is measured, not assumed: a probe on the projector output reads
    AUC 0.6094 -- the highest of any stage -- and an attention pool over the raw grid reaches
    0.6641 against 0.6001 for the mean+max summary every earlier number was based on.

    Applies only to per-label turns, identified by their verdict count (18 chest / 66
    abdomen, exact for all 2911 training answers). Local-VQA turns carry no per-label target
    and are skipped rather than given an invented one.
    """
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    w = float(os.environ["FLARE_AUX_CLS"])
    YES, NO = 10035, 912
    SPAN = {18: (0, 18), 66: (18, 84)}          # lineage -> its slice of the shared head

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        base = model.module if hasattr(model, "module") else model
        proj = base.get_model().mm_projector
        feat = getattr(proj, "_aux_feat", None)
        if not model.training or feat is None:
            return (out.loss, out) if return_outputs else out.loss
        labels = getattr(base, "_flare_expanded_labels", None)
        if labels is None:
            return (out.loss, out) if return_outputs else out.loss

        # keep the head's own dtype (the projector is bf16) and widen only the OUTPUT --
        # casting the input instead hits float32 activations against bf16 LayerNorm weights
        logits = proj.aux_head(feat).float()     # (B, 84)
        tgt = labels[:, 1:]
        rows, tgts = [], []
        for b in range(tgt.shape[0]):
            t = tgt[b]
            v = t[(t == YES) | (t == NO)]
            span = SPAN.get(int(v.numel()))
            if span is None:
                continue                          # local turn, or truncated: no target
            lo, hi = span
            rows.append(logits[b, lo:hi])
            tgts.append((v == YES).float())
        if not rows:
            return (out.loss, out) if return_outputs else out.loss
        aux = torch.stack([F.binary_cross_entropy_with_logits(r, t)
                           for r, t in zip(rows, tgts)]).mean()
        return ((out.loss + w * aux, out) if return_outputs
                else out.loss + w * aux)

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] aux label head on the projector ON, weight={w}", flush=True)


def patch_inject():
    """FLARE_INJECT: route the image into the answer WITHOUT crossing attention.

    Every objective tried so far had to send its gradient back through the attention the
    answer position pays to the image, and that path is measured at 2.224e-09 gradient per
    visual element against 2.770e-04 per text element -- a factor of 124,580. Widening the
    path did not help either: the image-last run doubled the attention on the image, from
    0.0277 to 0.0596, and the answer position stayed 99.5% cosine-identical across patients.

    So bypass it. A forward pre-hook on lm_head adds a learned projection of the pooled
    visual tokens into the residual stream at the point the logits are read, where no
    attention weight can attenuate it. Zero-initialised gate: the run starts as an exact
    copy of the trained model.

    The hook fires during generation too, and there the projector runs only on the first
    forward while later decode steps reuse the KV cache -- which is correct, since the image
    is the same for every step of one answer. The batch guard is there because a stale
    stash from a differently-sized batch would otherwise broadcast silently into the wrong
    rows instead of failing.
    """
    import torch
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    orig_init = LlavaLlamaForCausalLM.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._flare_inject_hooked = False

    def _install(self):
        if getattr(self, "_flare_inject_hooked", False):
            return
        proj = self.get_model().mm_projector
        if getattr(proj, "inject_head", None) is None:
            return

        def pre_hook(_mod, args):
            h = args[0]
            f = getattr(proj, "_aux_feat", None)
            if f is None or f.shape[0] != h.shape[0]:
                return None
            add = proj.inject_head(f.to(h.dtype)) * proj.inject_gate.to(h.dtype)
            return (h + add[:, None, :],)

        self.lm_head.register_forward_pre_hook(pre_hook)
        self._flare_inject_hooked = True
        print("[flare-patch] image injection hooked at lm_head (gate init 0)", flush=True)

    orig_fwd = LlavaLlamaForCausalLM.forward

    def forward(self, *a, **k):
        _install(self)
        return orig_fwd(self, *a, **k)

    LlavaLlamaForCausalLM.__init__ = __init__
    LlavaLlamaForCausalLM.forward = forward
    print("[flare-patch] FLARE_INJECT armed", flush=True)


def patch_align_loss():
    """FLARE_ALIGN=weight: put the visual tokens into the LLM's own vocabulary geometry.

    What is different from the aux head. FLARE_AUX_CLS hangs a learned Linear(3072->84) on
    the projector output and asks it to predict labels. That works -- but nothing in it
    requires the token to be something the LLM can read; the readout is arbitrary, and the
    run came out neutral (0.2627/0.4961 against 0.2645/0.4886).

    Here the label logits are cosine similarities against the LLM's OWN embeddings of the
    finding names:

        logit(label l) = scale * max_j cos( projector_token_j , embed("Lung nodule") ) + bias

    so the only way to lower the loss is to move visual tokens toward where the LLM already
    keeps that concept. The two scalars are the entire trainable surface -- every direction
    comes from the frozen embedding table, which is what stops this from quietly becoming
    another arbitrary readout.

    Contrast with the earlier projnorm run, which was the crude version of the same idea:
    it matched only the token NORM (47.55 -> 1.07) and made things worse, because forcing a
    single norm also erased the across-token magnitude differences and pushed cases back
    together (projector cross-case cosine 0.479 -> 0.685). Geometry, not scale.
    """
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    w = float(os.environ.get("FLARE_ALIGN", 0.0) or os.environ.get("FLARE_INFONCE", 0.0))
    wdiv = float(os.environ.get("FLARE_DIVERSITY", 0.0))
    # Measured on stage2_perlabel (diag_token_diversity.py, 12 cases): the 512 projector
    # tokens sit at mean pairwise cosine +0.886 WITHIN a case, against +0.044 on the raw
    # encoder grid they came from and +0.153 for a random sample of the LLM's own text
    # embeddings. Effective rank is 84 of 512. So the pooler collapses a near-orthogonal
    # grid into a handful of directions, and `max_j cos(token_j, embed(finding))` is then a
    # max over near-copies -- it cannot hand different findings to different tokens.
    #
    # The target is the TEXT embedding dispersion, not the grid's. The point is to look
    # like the space we are aligning into; driving the tokens to near-orthogonality would
    # be a different and unmotivated objective. Hinged, so tokens already spread past the
    # target are left alone.
    DIV_TARGET = 0.153
    wvar = float(os.environ.get("FLARE_VARIANCE", 0.0))
    # Rank-preserving term (VICReg-style variance hinge).
    #
    # The cosine hinge alone hit its target and still made things worse in a way the
    # earlier diagnostic caught: on BTB3D it drove within-case token cosine 0.886 -> 0.149
    # exactly as designed, while effective rank COLLAPSED 84 -> 14.6 of 512. Penalising
    # only the Gram off-diagonal admits a degenerate solution -- tokens split a handful of
    # axes by sign, so they look decorrelated while carrying less. "Spread out but saying
    # less."
    #
    # This constrains the other side: every embedding dimension must keep a standard
    # deviation across the token set. Hinged at 1.0 because the tokens are L2-normalised
    # per token before the statistic, so a healthy spread sits near that scale.
    VAR_TARGET = 1.0
    YES, NO = 10035, 912
    SPAN = {18: (0, 18), 66: (18, 84)}
    cache = {}

    def label_embeddings(model, tok):
        """One vector per finding name: mean of its token embeddings, frozen."""
        if "E" in cache:
            return cache["E"]
        import json as _json
        prior = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "data_clean/label_prior.json")))
        names = prior["chest"]["labels"] + prior["abd"]["labels"]
        emb = model.get_model().embed_tokens.weight
        vecs = []
        for n in names:
            ids = tok(n.replace("_", " "), add_special_tokens=False).input_ids
            vecs.append(emb[torch.tensor(ids, device=emb.device)].float().mean(0))
        E = F.normalize(torch.stack(vecs), dim=-1).detach()   # (84, 3072), never trained
        cache["E"] = E
        print(f"[flare-patch] label embeddings built for {len(names)} findings", flush=True)
        return E

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        base = model.module if hasattr(model, "module") else model
        proj = base.get_model().mm_projector
        toks = getattr(proj, "_tokens", None)
        labels = getattr(base, "_flare_expanded_labels", None)
        if not model.training or toks is None or labels is None:
            return (out.loss, out) if return_outputs else out.loss

        extra = 0.0
        if wdiv > 0:
            T0 = torch.nn.functional.normalize(toks.float(), dim=-1)
            G = torch.bmm(T0, T0.transpose(1, 2))
            n = G.shape[-1]
            off = (G.sum(dim=(1, 2)) - torch.diagonal(G, dim1=1, dim2=2).sum(-1)) / (n * (n - 1))
            extra = extra + wdiv * torch.relu(off - DIV_TARGET).mean()
        if wvar > 0:
            Tv = toks.float()
            Tv = Tv / Tv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            std = Tv.std(dim=1)                                  # (B, D) across tokens
            std = std * (Tv.shape[1] ** 0.5)                     # scale-free w.r.t. token count
            extra = extra + wvar * torch.relu(VAR_TARGET - std).mean()
        if w == 0:
            loss = out.loss + extra
            return (loss, out) if return_outputs else loss

        tk = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        E = label_embeddings(base, tk)
        T = F.normalize(toks.float(), dim=-1)                 # (B, 512, 3072)
        sim = torch.einsum("bnd,ld->bnl", T, E).amax(dim=1)   # (B, 84) best-matching token
        logits = proj.align_scale.float() * sim + proj.align_bias.float()

        tgt = labels[:, 1:]
        rows, tgts = [], []
        for b in range(tgt.shape[0]):
            v = tgt[b][(tgt[b] == YES) | (tgt[b] == NO)]
            span = SPAN.get(int(v.numel()))
            if span is None:
                continue                                       # local turn: no label vector
            lo, hi = span
            rows.append(logits[b, lo:hi]); tgts.append((v == YES).float())
        if not rows:
            return (out.loss, out) if return_outputs else out.loss
        if os.environ.get("FLARE_INFONCE"):
            # Within-case InfoNCE over the LABEL axis.
            #
            # BCE scores every label independently, which optimises calibration ACROSS
            # cases -- and across-case ranking is the one thing this pipeline already has
            # in quantity (feature AUC 0.7294 on held-out chest). The metric asks the other
            # question: within one patient, is finding A likelier than finding B? Measured,
            # our evidence loses that comparison to the base rate outright -- chest top-1
            # precision 0.340 against the prior's 0.497 -- which is why chest global keeps
            # landing on 0.2992, the constant answer, exactly.
            #
            # A softmax over the labels of a single case makes positives compete with that
            # case's own negatives, so the gradient is about ordering within the case
            # rather than about the label's overall rate. Multi-positive form: each present
            # finding is pulled above the same shared denominator.
            terms = []
            for r, t in zip(rows, tgts):
                if t.sum() == 0:
                    continue                    # no positive to rank; skip rather than
                                                # invent one
                logZ = torch.logsumexp(r, dim=0)
                terms.append(-((r - logZ) * t).sum() / t.sum())
            align = (torch.stack(terms).mean() if terms
                     else torch.zeros((), device=logits.device))
        else:
            align = torch.stack([F.binary_cross_entropy_with_logits(r, t)
                                 for r, t in zip(rows, tgts)]).mean()
        loss = out.loss + w * align + extra
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] alignment weight={w}  diversity weight={wdiv} "
          f"variance weight={wvar}  (target cosine {DIV_TARGET}, target std {VAR_TARGET})",
          flush=True)


def patch_asl_loss():
    """FLARE_ASL="gamma_neg,gamma_pos,margin": Asymmetric Loss at the yes/no verdicts.

    Asymmetric Loss for Multi-Label Classification (Ridnik et al., ICCV 2021) exists for
    exactly the shape this task has on the abdomen lineage: 66 candidate findings of which
    2.85 are present, i.e. 4.3% positives. Plain cross-entropy lets the 95.7% of easy
    negatives dominate the gradient, and the symptom is visible in our own numbers --
    abdominal P(yes) averages 0.037 with a per-case maximum of 0.30, so every probability
    is pressed toward zero and nothing separates.

    Two asymmetries, both from the paper:
      * focusing: down-weight easy negatives with (p)^gamma_neg while leaving positives
        nearly unfocused (gamma_pos = 0), because positives are the scarce signal.
      * probability shifting: a negative whose probability is already below `margin` is
        discarded outright, which is what stops a long tail of near-zero negatives from
        summing into a gradient larger than the positives'.

    Why it is worth a run HERE specifically, rather than as a generic "try another loss".
    The mechanism that actually earns score in this pipeline is the THRESHOLD rule --
    measured: adaptive cardinality from thresholding is worth +0.0458 on chest over the
    best fixed top-k, while a separately predicted count makes the metric monotonically
    WORSE (r 0.641 -> 0.666 and MAE 1.707 -> 1.468 took the score 0.3539 -> 0.3452). A
    threshold rule is only as good as the calibration of the probabilities it thresholds,
    so a loss that fixes calibration under extreme negative dominance acts directly on the
    part of the system that converts model quality into metric.

    Applied ONLY at verdict positions; every other token keeps cross-entropy, so the model
    still has to emit well-formed answers. Eval keeps the plain loss, so eval_loss stays
    comparable with every earlier run.
    """
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    gn, gp, margin = (float(x) for x in os.environ["FLARE_ASL"].split(","))
    w = float(os.environ.get("FLARE_ASL_W", "1.0"))
    YES, NO = 10035, 912                       # " yes" / " no" for this tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        if not model.training:
            return (out.loss, out) if return_outputs else out.loss
        base = model.module if hasattr(model, "module") else model
        labels = getattr(base, "_flare_expanded_labels", None)
        if labels is None:
            return (out.loss, out) if return_outputs else out.loss

        tgt = labels[:, 1:]
        sel = tgt != -100
        if not sel.any():
            return (out.loss, out) if return_outputs else out.loss
        logits = out.logits[:, :-1][sel].float()
        y = tgt[sel]
        verdict = (y == YES) | (y == NO)
        if not verdict.any():
            return (out.loss, out) if return_outputs else out.loss

        per = F.cross_entropy(logits, y, reduction="none")

        # binary probability of "yes" at the verdict positions
        vl = logits[verdict]
        p = torch.softmax(torch.stack([vl[:, YES], vl[:, NO]], dim=-1), dim=-1)[:, 0]
        pos = (y[verdict] == YES).float()
        eps = 1e-8
        pm = (p - margin).clamp(min=0)         # probability shifting, negatives only
        l_pos = ((1 - p) ** gp) * torch.log(p.clamp(min=eps))
        l_neg = (pm ** gn) * torch.log((1 - pm).clamp(min=eps))
        asl = -(pos * l_pos + (1 - pos) * l_neg)

        per = per.clone()
        per[verdict] = w * asl
        loss = per.mean()
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] ASL at verdict positions: gamma_neg={gn} gamma_pos={gp} "
          f"margin={margin} weight={w}", flush=True)


def patch_lsep_loss():
    """FLARE_LSEP=weight: log-sum-exp pairwise ranking over a case's own labels.

    Improving Pairwise Ranking for Multi-label Image Classification (Li et al., CVPR 2017).
    For one case with label scores f, positives P and negatives N:

        L = log( 1 + sum_{p in P} sum_{n in N} exp(f_n - f_p) )

    Every (positive, negative) pair inside ONE case is pushed apart, and the log-sum-exp
    keeps the loss smooth and bounded instead of letting one bad pair dominate the way a
    raw hinge sum does.

    Why this loss and not another. The metric is
        score = |hits| / max(|pred|, |gt|)
    and everything that earns score in this pipeline goes through the THRESHOLD rule, which
    is a WITHIN-CASE ranking followed by a cut. Measured on FLARE val: recall@5 is 0.605 on
    chest and 0.408 on abdomen against a 0.076 chance rate, so ranking signal exists -- but
    the model still lands on the constant-answer ceiling for abdomen because the ranking is
    not sharp enough at the top. Cross-entropy optimises each label independently and never
    once compares two labels of the same patient; LSEP optimises exactly that comparison.

    It is also the honest counterpart to ASL. ASL fixes CALIBRATION under 4.3% positives;
    LSEP fixes ORDER. They act on the two different halves of "rank, then cut", so running
    both separately says which half is actually short.

    Scores are the yes/no log-odds at the verdict positions, f_j = logit_yes - logit_no,
    which is the same quantity diag_perlabel_prob.py thresholds at inference -- so the loss
    optimises the number the rule actually reads, not a proxy for it.

    Added to the standard cross-entropy rather than replacing it: a pure ranking loss is
    invariant to a constant shift of f, and the threshold rule needs the absolute level too.
    """
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    w = float(os.environ["FLARE_LSEP"])
    YES, NO = 10035, 912                       # " yes" / " no" for this tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        if not model.training:
            return (out.loss, out) if return_outputs else out.loss
        base = model.module if hasattr(model, "module") else model
        labels = getattr(base, "_flare_expanded_labels", None)
        if labels is None:
            return (out.loss, out) if return_outputs else out.loss

        tgt = labels[:, 1:]
        sel = tgt != -100
        if not sel.any():
            return (out.loss, out) if return_outputs else out.loss
        logits = out.logits[:, :-1]

        terms = []
        for b in range(tgt.shape[0]):
            m = sel[b]
            if not m.any():
                continue
            t = tgt[b][m]
            v = (t == YES) | (t == NO)
            if v.sum() < 2:
                continue
            lg = logits[b][m][v].float()
            f = lg[:, YES] - lg[:, NO]         # the log-odds the threshold rule reads
            pos = f[t[v] == YES]
            neg = f[t[v] == NO]
            if pos.numel() == 0 or neg.numel() == 0:
                continue                       # no pair to rank; CE still supervises it
            d = neg[None, :] - pos[:, None]    # (|P|, |N|)
            terms.append(torch.log1p(torch.exp(d.clamp(max=30)).sum()))

        if not terms:
            return (out.loss, out) if return_outputs else out.loss
        loss = out.loss + w * torch.stack(terms).mean()
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] LSEP pairwise ranking ON, weight={w}", flush=True)


def patch_rank_head_loss():
    """FLARE_RANK_HEAD=weight: LSEP on a separate head over the answer-position states.

    Same ranking objective as patch_lsep_loss, moved off the LM head. The score for label j
    is rank_head(h_j) where h_j is the last hidden state at that label's verdict position --
    so the ranking gradient reaches the transformer and the projector, but never the output
    embedding matrix that has to keep producing grammatical text.

    Cross-entropy is unchanged and still the main objective; this is added on top, so the
    model that generates answers is the same model, plus a head that can be read instead of
    p_yes at inference.
    """
    import torch
    from llava.train.llava_trainer import LLaVATrainer

    w = float(os.environ["FLARE_RANK_HEAD"])
    YES, NO = 10035, 912
    # FLARE_RANK_HEAD_DOM restricts the ranking term to one lineage. This is a LOSS mask,
    # not a second model: one checkpoint comes out, trained with a term that is zero on the
    # samples it does not apply to -- the same shape as a class-weighted loss.
    #
    # Worth having because the measured effect is one-sided. On the pooler base the rank
    # head moved abdomen +0.0125 (95% CI [+0.0037, +0.0215], past the 0.2377 constant-answer
    # ceiling for the first time in this project) while chest went -0.0050, inside noise.
    #
    # Lineage is read off the number of verdict positions -- 18 chest, 66 abdomen, verified
    # exact for all 2,911 training answers -- the same test patch_lmh_loss uses.
    dom = (os.environ.get("FLARE_RANK_HEAD_DOM") or "").lower()
    assert dom in ("", "chest", "abd"), f"FLARE_RANK_HEAD_DOM={dom!r}; expected chest|abd"
    N_VERDICT = {"chest": 18, "abd": 66}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        base = model.module if hasattr(model, "module") else model
        if not model.training:
            out = model(**inputs)
            return (out.loss, out) if return_outputs else out.loss
        out = model(**inputs, output_hidden_states=True)
        labels = getattr(base, "_flare_expanded_labels", None)
        head = base.get_model().mm_projector.rank_head
        if labels is None or head is None or out.hidden_states is None:
            return (out.loss, out) if return_outputs else out.loss

        h = out.hidden_states[-1]                 # (B, T, d) -- includes visual positions
        tgt = labels[:, 1:]
        terms = []
        for b in range(tgt.shape[0]):
            m = tgt[b] != -100
            if not m.any():
                continue
            t = tgt[b][m]
            v = (t == YES) | (t == NO)
            if v.sum() < 2:
                continue
            if dom and int(v.sum()) != N_VERDICT[dom]:
                continue                    # other lineage: this term contributes nothing
            # the state that PRODUCED the verdict token sits one position earlier, which is
            # also the position whose probe reads AUC 0.7415
            idx = m.nonzero(as_tuple=True)[0][v]
            # the head lives in the model's dtype (bf16); cast the OUTPUT, not the input,
            # or the LayerNorm gets float32 activations against bf16 weights
            f = head(h[b][idx]).squeeze(-1).float()
            pos, neg = f[t[v] == YES], f[t[v] == NO]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            d = neg[None, :] - pos[:, None]
            terms.append(torch.log1p(torch.exp(d.clamp(max=30)).sum()))

        if not terms:
            return (out.loss, out) if return_outputs else out.loss
        loss = out.loss + w * torch.stack(terms).mean()
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] LSEP on a separate answer-state head, weight={w}"
          f"{', ' + dom + ' only' if dom else ''}", flush=True)


def patch_text_contrast_loss():
    """FLARE_TXTCON=weight: contrast each label's ANSWER-POSITION hidden state against that
    label's own text, within the case.

    Three text-alignment objectives have already failed here -- FLARE_ALIGN (0.2646),
    aligndiv (0.2624), InfoNCE (0.2629), all inside noise of the 0.2654 baseline. Every one
    of them acted on the VISUAL tokens. The measurements say that is the wrong place: the
    projector's tokens are already distinct across patients (same-position cross-case cosine
    0.7257 against the encoder grid's 0.4965), while the answer position is not (0.9999).
    Whatever those losses shaped, the LLM discarded before the answer.

    The rank head moved the same axis those losses could not: abdomen +0.0125, CI
    [+0.0037, +0.0215], past the constant-answer ceiling for the first time. Its one
    structural difference is WHERE it attaches -- a head on the answer-position hidden state.
    This puts a text signal at that same place.

        score_j = cos( proj(h_j), e_j ) / tau        e_j = embedding of label j's text
        loss    = LSEP over the case's own labels, positives above negatives

    So it is the same within-case ranking objective as the rank head, except the score comes
    from agreement with the label's TEXT rather than from a free scalar head. If it beats
    the rank head, the text is carrying information the scalar cannot; if it ties, the
    ranking pressure was the whole story and text granularity is a dead end here.

    FLARE_TXTCON_FORM=sentence embeds "There is <label>." instead of the bare name -- the
    coarsest step toward the per-lesion sentence granularity, without needing per-case report
    text (which the collator does not carry a case id for).

    Label identity comes from the verdict position: 18 chest / 66 abdomen, the i-th verdict
    is the i-th condition in data_clean/label_prior.json, verified exact for all 2,911
    training answers. A sample matching neither count is skipped rather than guessed at.
    """
    import json as _json
    import torch
    import torch.nn.functional as F
    from llava.train.llava_trainer import LLaVATrainer

    w = float(os.environ["FLARE_TXTCON"])
    form = (os.environ.get("FLARE_TXTCON_FORM") or "name").lower()
    tau = float(os.environ.get("FLARE_TXTCON_TAU", "0.07"))
    YES, NO = 10035, 912
    prior = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data_clean/label_prior.json")))
    by_len = {len(v["labels"]): v["labels"] for v in prior.values()}
    cache = {}

    def embeds(model, tok, n):
        """mean token embedding of each label's text, L2-normalised. Built once."""
        if n in cache:
            return cache[n]
        emb = model.get_model().embed_tokens
        rows = []
        for name in by_len[n]:
            t = name.replace("_", " ")
            if form == "sentence":
                t = f"There is {t}."
            ids = torch.tensor(tok.encode(t, add_special_tokens=False), device=emb.weight.device)
            rows.append(emb(ids).float().mean(0))
        # DETACHED, for two reasons. Correctness: the label text is the fixed target the
        # hidden state should move toward -- letting gradient into the embedding table would
        # let the target drift to meet the state, which is the degenerate solution. And
        # mechanically: these are cached across batches, so a tensor still attached to the
        # graph makes the second step die with "Trying to backward through the graph a
        # second time" once the first backward has freed it.
        e = F.normalize(torch.stack(rows), dim=-1).detach()
        cache[n] = e
        return e

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        base = model.module if hasattr(model, "module") else model
        if not model.training:
            out = model(**inputs)
            return (out.loss, out) if return_outputs else out.loss
        out = model(**inputs, output_hidden_states=True)
        labels = getattr(base, "_flare_expanded_labels", None)
        proj = getattr(base.get_model().mm_projector, "txt_proj", None)
        if labels is None or proj is None or out.hidden_states is None:
            return (out.loss, out) if return_outputs else out.loss

        h = out.hidden_states[-1]
        tgt = labels[:, 1:]
        terms = []
        for b in range(tgt.shape[0]):
            m = tgt[b] != -100
            if not m.any():
                continue
            t = tgt[b][m]
            v = (t == YES) | (t == NO)
            n = int(v.sum())
            if n not in by_len:
                continue
            idx = m.nonzero(as_tuple=True)[0][v]
            z = F.normalize(proj(h[b][idx]).float(), dim=-1)
            e = embeds(base, self.tokenizer, n)
            s = (z * e).sum(-1) / tau
            pos, neg = s[t[v] == YES], s[t[v] == NO]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            d = neg[None, :] - pos[:, None]
            terms.append(torch.log1p(torch.exp(d.clamp(max=30)).sum()))

        if not terms:
            return (out.loss, out) if return_outputs else out.loss
        loss = out.loss + w * torch.stack(terms).mean()
        return (loss, out) if return_outputs else loss

    LLaVATrainer.compute_loss = compute_loss
    print(f"[flare-patch] text contrast at the answer states: weight={w} form={form} "
          f"tau={tau}", flush=True)


if __name__ == "__main__":
    patch_vision_modules()
    patch_projector()
    patch_eval_dataset()
    if os.environ.get("FLARE_IMAGE_POS") == "back":
        patch_image_position()
    if os.environ.get("FLARE_LMH_LAMBDA"):
        patch_capture_labels()
        patch_lmh_loss()
    if os.environ.get("FLARE_AUX_CLS"):
        patch_capture_labels()
        patch_aux_cls_loss()
    if os.environ.get("FLARE_TXTCON"):
        patch_capture_labels()
        patch_text_contrast_loss()
    if os.environ.get("FLARE_RANK_HEAD"):
        patch_capture_labels()
        patch_rank_head_loss()
    if os.environ.get("FLARE_LSEP"):
        patch_capture_labels()
        patch_lsep_loss()
    if os.environ.get("FLARE_ASL"):
        patch_capture_labels()
        patch_asl_loss()
    if os.environ.get("FLARE_INJECT"):
        patch_inject()
    if (os.environ.get("FLARE_ALIGN") or os.environ.get("FLARE_DIVERSITY")
            or os.environ.get("FLARE_VARIANCE") or os.environ.get("FLARE_INFONCE")):
        patch_capture_labels()
        patch_align_loss()
    from llava.train.train import train
    train(attn_implementation="sdpa")
