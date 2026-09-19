"""AttentionalPoolProjector with pooling re-enabled, n_queries configurable.

Upstream (`BTB3D/report-generation/llava/model/multimodal_projector/coca_attentional_pooler.py`)
has the attention-pool call commented out ("remove attentionalpool for now"), so its
"attn_pool+mlp2x_gelu" projector actually just flattens+MLPs every voxel-code token --
31,744 tokens for the 8_8_8 (folded) encoder. That is what made BTB3D-8's zero-shot
report generation slow enough to risk the FLARE runtime budget.

The class is otherwise copied verbatim, including `AttentionalPooler` (already
implemented and unused upstream). This file lives in our own training dir rather than
patching the shared BTB3D checkout, so other people's pipelines are untouched.

Dimension note: the checkpoint that trained cleanly (llava_btb3d_8_lora_ckpt) has BOTH
mm_hidden_size=72 AND mm_context_size=72 in its saved config -- llava_arch.py's
`self.config.mm_context_size = 18` hardcode is stale/overridden dead code for the 8_8_8
variant. We therefore set embed_dim=context_dim=72 explicitly rather than trusting that
hardcode.
"""
import os
from typing import Callable

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


def embed_dim_out(projector):
    """Output width of the MLP handed in as `projector` -- its last Linear's out_features."""
    for m in reversed(list(projector.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("projector has no Linear layer to read an output width from")


class AttentionalPooler(nn.Module):
    """Copied from coca_attentional_pooler.py, unchanged."""

    def __init__(self, d_model: int, context_dim: int, n_head: int = 8,
                n_queries: int = 512, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        dim_head = d_model // n_head
        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head
        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)
        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = rearrange(x, "b n d -> b 1 n d")
        q = repeat(self.query, "n d -> b m n d", b=x.shape[0], m=x.shape[1])
        x = self.ln_k(x)
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(q)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        sim = einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out).squeeze(dim=1)


class AttentionalPoolProjectorEnabled(nn.Module):
    """Pooling restored: 512 learned queries attend over every voxel-code token,
    collapsing (T, H, W) -> 512 BEFORE the MLP projects to the LLM hidden size.
    31,744 -> 512 tokens for the 8_8_8 encoder, matching Med3DVLM/M3D-LaMed's budget
    order of magnitude instead of sitting ~2 orders of magnitude above it.
    """

    def __init__(self, embed_dim, context_dim, projector=None, n_head=8, n_queries=512,
                norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.attn_pool = AttentionalPooler(d_model=embed_dim, context_dim=context_dim,
                                           n_head=n_head, n_queries=n_queries,
                                           norm_layer=norm_layer)
        self.ln = norm_layer(embed_dim)
        self.proj = projector if projector else nn.Identity()
        # FLARE_PROJ_NORM is the target token norm; unset keeps the original behaviour so
        # every checkpoint trained before this measurement still loads and scores the same.
        # Stored as a plain float, not a buffer, so state_dicts stay compatible both ways.
        tgt = os.environ.get("FLARE_PROJ_NORM")
        self.out_scale = None if not tgt else float(tgt) / (embed_dim_out(projector) ** 0.5)
        # FLARE_AUX_CLS: a label head straight on the pooled visual tokens.
        #
        # Measured: the gradient the LLM loss delivers to the visual features is 2.2e-09
        # per element against 2.8e-04 on the text embeddings -- 124,580x smaller -- because
        # everything reaching the projector must pass back through the attention the answer
        # position pays to the image, which is 0.028. Three objectives (contrastive
        # decoding, mDPO, product-of-experts debiasing) stalled on exactly that factor.
        #
        # This head sits BEFORE the bottleneck, so its gradient never crosses it. And there
        # is signal here to supervise: a probe on these features reads AUC 0.6094, the best
        # of any stage in the pipe, and an attention pool over the raw grid reaches 0.6641.
        #
        # 84 = 18 chest labels then 66 abdomen; a sample uses only its lineage's slice, so
        # one head serves both without letting either lineage's gradient touch the other's.
        self.aux_head = None
        if os.environ.get("FLARE_AUX_CLS"):
            d = embed_dim_out(projector)
            self.aux_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 84))
        # FLARE_INJECT: a second, non-competing route from the image to the answer.
        #
        # In a transformer the ONLY way information crosses positions is attention, and the
        # answer position spends 0.028 of its attention on the image block that occupies
        # 0.725 of the sequence. Putting the image next to the answer doubled that to 0.0596
        # and changed nothing: the answer position's hidden state stayed 99.5% cosine-
        # identical across patients, so the loss is not attention mass alone.
        #
        # This adds the pooled image straight into the residual stream just before the
        # output head, which no attention weight can attenuate. The gate starts at zero so
        # the run begins as an exact copy of the trained model and the injection has to earn
        # its way in, rather than perturbing a working model on step one.
        self.inject_head = None
        if os.environ.get("FLARE_INJECT"):
            d = embed_dim_out(projector)
            self.inject_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            # The first version initialised this gate at 0, ReZero/LayerScale style, so the
            # run would begin as an exact copy of the trained model. That is a dead start
            # here, not a safe one: the injection enters as gate * head(image), so
            #     dL/d(head weights)  ~  gate
            # and with gate == 0 the head receives NO gradient at all, while the gate itself
            # only gets gradient through a randomly initialised head -- i.e. noise. Both
            # sit at a saddle. It ended 3 epochs at +0.000047, which measures that saddle
            # and says nothing about whether the image is useful.
            #
            # So the gate starts at 1 and the SCALE is moved into the head's last layer
            # instead: the injection is still small on step one, but gate and head both
            # receive real gradient from the first step.
            self.inject_gate = nn.Parameter(torch.ones(1))
            with torch.no_grad():
                self.inject_head[1].weight.mul_(0.01)
                self.inject_head[1].bias.zero_()
        # FLARE_TXTCON: projection from the LLM's answer-state space into the space its
        # own token embeddings live in, so the two can be compared by cosine. Lives on the
        # projector for the same reason rank_head does -- non_lora_trainables saves it.
        self.txt_proj = None
        if os.environ.get("FLARE_TXTCON"):
            d = embed_dim_out(projector)
            self.txt_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self._aux_feat = None
        self._tokens = None
        if os.environ.get("FLARE_ALIGN") or os.environ.get("FLARE_DIVERSITY") or os.environ.get("FLARE_VARIANCE"):
            # two scalars only. The DIRECTION of every label logit comes from the LLM's own
            # token embeddings, not from learned weights -- that is the whole point, so the
            # trainable surface here is a temperature and a bias, nothing that could learn
            # the task on its own.
            # shape (1,), not 0-dim: transformers' from_pretrained does
            # torch.empty(*param.size(), ...) and a 0-dim parameter expands to
            # torch.empty() -- "missing 1 required positional argument: size". Training
            # survives it (deepspeed takes another path) but the checkpoint then cannot be
            # loaded for inference, so the run would only fail at scoring time.
            self.align_scale = nn.Parameter(torch.tensor([14.0]))
            self.align_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        # x: (B, T, H, W, D) voxel-code grid from the MAGViT2 encoder
        B, X_dim, Y, Z, D = x.shape
        x = x.flatten(1, 3)              # (B, T*H*W, D)
        tokens = self.attn_pool(x)       # (B, n_queries, D)
        tokens = self.ln(tokens)
        tokens = self.proj(tokens)       # (B, n_queries, hidden_size)
        if self.out_scale is not None:
            # Force every visual token onto the same norm as a text embedding.
            # Measured (diag_signal_path.py, stage2_perlabel): the projector emits tokens
            # of norm 47.55 next to text embeddings of norm 1.07 -- 44x -- because `ln`
            # above sits BEFORE `proj` and nothing normalises what `proj` emits. Those
            # tokens land far outside the distribution the pretrained layers were
            # calibrated on, and the last prompt token ends up spending 2.8% of its
            # attention on the 72.5% of the sequence they occupy.
            #
            # Deliberately NOT learnable and applied after `proj`: the projector trains
            # during Stage 2, so a learnable scale could simply drift back up and the
            # run would stop being a test of anything.
            rms = tokens.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            tokens = tokens * rms * self.out_scale
        if os.environ.get("FLARE_ALIGN") or os.environ.get("FLARE_DIVERSITY") or os.environ.get("FLARE_VARIANCE"):
            # full token set, for the alignment and diversity losses in launch_train.py.
            # Gated on BOTH: a diversity-only run would otherwise find _tokens None and the
            # loss would silently do nothing.
            self._tokens = tokens
        if self.aux_head is not None or self.inject_head is not None:
            # stashed rather than returned: llava_arch splices whatever forward() returns
            # into the sequence, so an extra return value would corrupt the prompt
            self._aux_feat = tokens.mean(dim=1)
        return tokens


def build_pooled_projector(config, n_queries=512):
    """config needs mm_hidden_size, mm_context_size, hidden_size (all 72/72/3072 for
    BTB3D-8 -> Llama-3.2-3B)."""
    mlp = nn.Sequential(
        nn.Linear(config.mm_hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
    )
    return AttentionalPoolProjectorEnabled(
        embed_dim=config.mm_hidden_size,
        context_dim=config.mm_context_size,
        projector=mlp,
        n_queries=n_queries,
    )
