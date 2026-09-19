"""Position-preserving projectors for the Med3DVLM (DCFormer) features.

Why this file exists
--------------------
The incumbent is an AttentionalPooler: n learned query vectors, softmax cross-attention
over the encoder tokens. Measured (diag_projector_candidates.py, 48 FLARE val cases):

    within-case token cosine        effective rank
    pooler q512 <- 32tok  TRAINED   +0.9950     89.1 / 512
    pooler q32  <- 288tok TRAINED   +0.9970     25.7 / 32
    pooler q512 <- 32tok  at init   +0.9893     91.0 / 512     <-- untrained, same collapse
    input grid (low, 256 tok)       +0.5245

The at-init row is the load-bearing one. The collapse is NOT something training drives the
pooler into; a randomly initialised pooler already emits one vector n times. The mechanism
is that out_i = sum_j softmax(q_i.k_j) v_j, and when the q.k logits have little spread every
query's softmax is near-uniform, so every output is approximately mean_j(v_j) -- independent
of i. Training moved it 0.9893 -> 0.9950, i.e. essentially nowhere.

So this is structural, and it is inherited by ANY learned-query softmax pooler. The fix in
the literature is to make output token k depend on a FIXED region of the input grid rather
than on a data-dependent softmax:

    adapt   DeCo (ICLR 2025)        parameter-free 3D adaptive average pooling
    shuffle InternVL / LaCo         fold a 2x2x1 neighbourhood into the channel axis
    cabs    Honeybee (CVPR 2024)    local 3D convs -> adaptive pool -> local 3D convs
    mixer   Med3DVLM's own          dual-stream MLP-Mixer; token mixing is a fixed learned
                                    linear map over POSITIONS, so its weights cannot depend
                                    on the patient and two patients cannot be mapped onto
                                    the same vector by an input-dependent attention

Grid layout (verified against the encoder itself in diag_grid_layout.py, not inferred from
the token count -- 256 also factors as 16x4x4 and the wrong choice scrambles every conv):

    low  stream  (T,H,W) = (4,8,8)  x 384 ch  -> 256 tokens
    high stream  (T,H,W) = (2,4,4)  x 768 ch  ->  32 tokens

The stored .npz files hold 288 tokens x 768 where tokens[:256] are the low stream with its
384 channels TILED TWICE (verified exact, max|diff| = 0), so both streams are recoverable
here with no re-encoding.
"""
from __future__ import annotations
import os

import torch
import torch.nn as nn

LOW_THW = (4, 8, 8)
LOW_C = 384
HIGH_THW = (2, 4, 4)
HIGH_C = 768
N_LOW = LOW_THW[0] * LOW_THW[1] * LOW_THW[2]     # 256
N_HIGH = HIGH_THW[0] * HIGH_THW[1] * HIGH_THW[2]  # 32


def _need_low(lo, name):
    """A two-stream projector handed high-only features would otherwise die on
    `NoneType.flatten` several frames deep. Say which projector and which directory."""
    if lo is None:
        raise ValueError(
            f"FLARE_PROJ={name} needs the low stream; point --image_folder at a "
            f"{N_LOW + N_HIGH}-token directory (m3d2_feat_*), not a high-only one")
    return lo


def _split(x):
    """llava hands over (B, n, 1, 1, 768) -> low (B,384,T,H,W), high (B,768,T,H,W).

    Two accepted layouts, and nothing else -- an unrecognised token count would otherwise
    be reshaped into a grid that does not exist and train silently on scrambled space:

      n = 288  the two-stream files (m3d2_feat_*): 256 low tokens with their 384 channels
               tiled twice, then 32 high tokens.
      n =  32  the high-only files (m3d_feat_*): no low stream exists, so `low` is None and
               a projector that needs it must say so.

    The 32-token path matters because that lineage is the one that actually beats the
    constant-answer ceiling (chest 0.3142 vs 0.2992); the 288-token files regressed to
    exactly the ceiling with BOTH projectors, so the projector question has to be asked
    again on top of the features that work.
    """
    B = x.shape[0]
    t = x.reshape(B, -1, x.shape[-1])
    n = t.shape[1]
    if n == N_LOW + N_HIGH:
        lo = t[:, :N_LOW, :LOW_C].reshape(B, *LOW_THW, LOW_C).permute(0, 4, 1, 2, 3)
        hi = t[:, N_LOW:, :].reshape(B, *HIGH_THW, HIGH_C).permute(0, 4, 1, 2, 3)
        return lo.contiguous(), hi.contiguous()
    if n == N_HIGH:
        hi = t.reshape(B, *HIGH_THW, HIGH_C).permute(0, 4, 1, 2, 3)
        return None, hi.contiguous()
    raise ValueError(
        f"spatial projector accepts {N_LOW + N_HIGH}-token (two-stream) or {N_HIGH}-token "
        f"(high-only) features, got {n} tokens per case")


def _img_drop(z, training):
    """Drop visual tokens with probability p during training (FLARE_IMG_DROP).

    Every axis that has worked so far operates by blocking the text shortcut so the model
    has to use the image; region-wise supervision and question dropout reach the same
    ceiling by different routes. This is the same family from the opposite direction: make
    the visual tokens incomplete so the model has to extract more from the ones that remain.
    """
    import os as _o
    p = float(_o.environ.get("FLARE_IMG_DROP", "0") or 0)
    if not training or p <= 0:
        return z
    keep = (torch.rand(z.shape[0], z.shape[1], 1, device=z.device) >= p).to(z.dtype)
    return z * keep / max(1e-6, 1.0 - p)


class DirectProjector(nn.Module):
    """The minimal change from the AttentionalPooler, and the cleanest test of DeCo's claim.

    Same 32 tokens in, same 32 tokens out. No learned queries, no softmax, no compression:
    output token k IS grid position k, lifted by an MLP. Everything else about the run is
    held fixed, so whatever moves is attributable to the mixing mechanism alone -- which is
    not true of any arm built on the 288-token files, where the feature format changed too.
    """

    def __init__(self, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(HIGH_C)
        self.proj = _mlp(HIGH_C, hidden)

    def forward(self, x):
        _, hi = _split(x)
        h = hi.flatten(2).transpose(1, 2)          # (B, 32, 768)
        return _img_drop(self.proj(self.ln(h)), self.training)


def _mlp(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout), nn.GELU(), nn.Linear(cout, cout))


class DirectMWProjector(nn.Module):
    """Lift 96 multi-window tokens (32 per window x 3 windows) without pooling.

    `mw_feat_*` encodes the lung, abdominal, and bone windows separately and stacks them
    along the token axis. Three chest labels -- `arterial wall calcification`,
    `coronary artery wall calcification`, and `hiatal hernia` -- are not discernible in the
    lung window (-1350 to 150 HU), so single-window features do not contain that information
    at all.

    Channel statistics differ per window, so each window gets its own LayerNorm; sharing one
    lets the highest-contrast window dominate the rest.
    """

    def __init__(self, hidden, n_win=3):
        super().__init__()
        self.n_win = n_win
        self.ln = nn.ModuleList([nn.LayerNorm(HIGH_C) for _ in range(n_win)])
        self.proj = _mlp(HIGH_C, hidden)

    def forward(self, x):
        # The projector receives (B, N, 1, 1, C): token axis first, channels last. Assuming
        # (B, N, C) and inserting a transpose produced (4, 256, 96) and broke LayerNorm.
        # Do not permute the axes -- only collapse the trailing ones.
        t = x.reshape(x.shape[0], x.shape[1], -1) if x.dim() > 3 else x
        B, N, C = t.shape
        assert C == HIGH_C, f"got {C} channels, expected {HIGH_C}"
        assert N % self.n_win == 0, f"{N} tokens do not divide into {self.n_win} windows"
        per = N // self.n_win
        outs = [self.proj(self.ln[i](t[:, i * per:(i + 1) * per, :]))
                for i in range(self.n_win)]
        return torch.cat(outs, dim=1)


class DirectBothProjector(nn.Module):
    """Lift all 288 tokens (256 low-resolution + 32 high-resolution) without pooling.

    `direct` uses only the 32 high-resolution tokens. The 288-token lineage had so far been
    tried only with an AttentionalPooler, a Mixer, and a C-Abstractor, all three of which sat
    at the constant-answer ceiling -- and all three mix positions together. No variant had
    lifted all 288 while preserving position, which is worth testing because root questions
    name a location ("the left kidney"), so resolution may be the bottleneck.

    The two streams have different channel counts (384 and 768), so each is lifted separately
    and the results are concatenated along the token axis.
    """

    def __init__(self, hidden):
        super().__init__()
        self.ln_lo = nn.LayerNorm(LOW_C)
        self.ln_hi = nn.LayerNorm(HIGH_C)
        self.p_lo = _mlp(LOW_C, hidden)
        self.p_hi = _mlp(HIGH_C, hidden)

    def forward(self, x):
        lo, hi = _split(x)
        a = self.p_lo(self.ln_lo(lo.flatten(2).transpose(1, 2)))
        b = self.p_hi(self.ln_hi(hi.flatten(2).transpose(1, 2)))
        return torch.cat([a, b], dim=1)


class AdaptPoolProjector(nn.Module):
    """DeCo. Compression is positional and parameter-free; only the lift is learned."""

    def __init__(self, hidden, out_thw=(2, 4, 4), use_high=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(out_thw)
        self.low = _mlp(LOW_C, hidden)
        self.use_high = use_high
        self.high = _mlp(HIGH_C, hidden) if use_high else None

    def forward(self, x):
        lo, hi = _split(x)
        lo = _need_low(lo, "adapt")
        l = self.pool(lo).flatten(2).transpose(1, 2)
        out = self.low(l)
        if self.use_high:
            h = hi.flatten(2).transpose(1, 2)
            out = torch.cat([out, self.high(h)], dim=1)
        return out


class PixelShuffleProjector(nn.Module):
    """InternVL / LaCo. 2x2x1 in-plane fold: no averaging, so nothing is thrown away."""

    def __init__(self, hidden, use_high=True):
        super().__init__()
        self.low = _mlp(LOW_C * 4, hidden)
        self.use_high = use_high
        self.high = _mlp(HIGH_C, hidden) if use_high else None

    def forward(self, x):
        lo, hi = _split(x)
        lo = _need_low(lo, "shuffle")
        B, C, T, H, W = lo.shape
        l = lo.permute(0, 2, 3, 4, 1)                       # (B,T,H,W,C)
        l = l.reshape(B, T, H // 2, 2, W // 2, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        l = l.reshape(B, T * (H // 2) * (W // 2), 4 * C)    # (B,64,1536)
        out = self.low(l)
        if self.use_high:
            h = hi.flatten(2).transpose(1, 2)
            out = torch.cat([out, self.high(h)], dim=1)
        return out


class CAbstractor3D(nn.Module):
    """Honeybee's C-Abstractor in 3D. The convs mix only neighbours, so locality survives
    the compression instead of being decided by a softmax."""

    def __init__(self, hidden, cmid=512, out_thw=(2, 4, 4), n_blocks=3, use_high=True):
        super().__init__()

        def blk(c):
            return nn.Sequential(nn.Conv3d(c, c, 3, padding=1),
                                 nn.GroupNorm(8, c), nn.GELU())

        self.inp = nn.Conv3d(LOW_C, cmid, 1)
        self.pre = nn.Sequential(*[blk(cmid) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool3d(out_thw)
        self.post = nn.Sequential(*[blk(cmid) for _ in range(n_blocks)])
        self.low = _mlp(cmid, hidden)
        self.use_high = use_high
        self.high = _mlp(HIGH_C, hidden) if use_high else None

    def forward(self, x):
        lo, hi = _split(x)
        lo = _need_low(lo, "cabs")
        l = self.post(self.pool(self.pre(self.inp(lo))))
        out = self.low(l.flatten(2).transpose(1, 2))
        if self.use_high:
            h = hi.flatten(2).transpose(1, 2)
            out = torch.cat([out, self.high(h)], dim=1)
        return out


class MixerProjector(nn.Module):
    """Med3DVLM's own projector, imported from the baseline rather than reimplemented.

    Their Table 9 ablation changes ONLY this module: 2xMLP 15.10 -> 1xMixer 23.25 ->
    2xMixer-H 36.42 METEOR. It is the projector DCFormer was trained alongside, and using
    it keeps us inside one model's components -- no second encoder is involved.

    Output is 128 low + 128 high = 256 tokens, i.e. FEWER than the 512 the pooler emitted.
    """

    def __init__(self, hidden):
        super().__init__()
        import os, sys
        MED = os.environ.get("MED3DVLM_DIR")
        if not MED:
            raise RuntimeError("set MED3DVLM_DIR to a checkout of https://github.com/mirthAI/Med3DVLM")
        if MED not in sys.path:
            sys.path.insert(0, MED)
        from src.model.projector.mlp import MixerLowHighHybridMLP
        assert hidden % 4 == 0, "MixerLowHighHybridMLP needs hidden divisible by 2**depth"
        self.m = MixerLowHighHybridMLP(
            low_input_size=(N_LOW, LOW_C), low_output_size=[192, 128],
            high_input_size=(N_HIGH, HIGH_C), high_output_size=[64, 128],
            output_dim=hidden, depth=2, mlp_depth=2)

    def forward(self, x):
        lo, hi = _split(x)
        lo = _need_low(lo, "mixer")
        l = lo.flatten(2).transpose(1, 2)      # (B,256,384)
        h = hi.flatten(2).transpose(1, 2)      # (B,32,768)
        return self.m((l, h))


def _attach_rank_head(m, hidden):
    """FLARE_RANK_HEAD needs a head on every projector, not just the pooler.

    patch_rank_head_loss reads `mm_projector.rank_head`, and it lives on the projector so
    non_lora_trainables saves it with the rest. The pooler grew one in attn_pool_projector;
    without this the spatial projectors do not, and the run dies on the first batch with
    "'DirectProjector' object has no attribute 'rank_head'".
    """
    import os as _os
    if _os.environ.get("FLARE_RANK_HEAD"):
        m.rank_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    else:
        m.rank_head = None
    if _os.environ.get("FLARE_TXTCON"):
        m.txt_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
    else:
        m.txt_proj = None
    return m


def build_spatial_projector(kind, hidden):
    kind = kind.lower()
    if kind == "mixer":
        return _attach_rank_head(MixerProjector(hidden), hidden)
    if kind == "adapt":
        return _attach_rank_head(AdaptPoolProjector(hidden), hidden)
    if kind == "adapt256":
        return _attach_rank_head(AdaptPoolProjector(hidden, out_thw=LOW_THW), hidden)
    if kind == "shuffle":
        return _attach_rank_head(PixelShuffleProjector(hidden), hidden)
    if kind == "cabs":
        return _attach_rank_head(CAbstractor3D(hidden), hidden)
    if kind == "direct":
        return _attach_rank_head(DirectProjector(hidden), hidden)
    if kind == "directmw":
        return _attach_rank_head(DirectMWProjector(hidden), hidden)
    if kind == "directboth":
        return _attach_rank_head(DirectBothProjector(hidden), hidden)
    raise ValueError(f"unknown FLARE_PROJ={kind!r}; expected "
                     "mixer | adapt | adapt256 | shuffle | cabs | direct | directboth | directmw")


def install_projector_patch(n_queries):
    """Point llava's builder at whichever projector FLARE_PROJ selects.

    Training and inference each used to carry their OWN copy of this monkeypatch, and that
    duplication is exactly what broke arm A's scoring: FLARE_PROJ was added to
    launch_train.py only, so the eval rebuilt an AttentionalPooler and refused to load a
    mixer checkpoint. One function, called from both, so the two cannot drift again.

    Returns nothing; unset FLARE_PROJ keeps the AttentionalPooler, which is what every
    pre-existing checkpoint and script needs.
    """
    import os as _os
    from attn_pool_projector import AttentionalPoolProjectorEnabled, embed_dim_out
    import llava.model.multimodal_projector.builder as builder_mod

    kind = _os.environ.get("FLARE_PROJ")

    class _Bound(AttentionalPoolProjectorEnabled):
        def __init__(self, embed_dim, context_dim, projector=None, **kw):
            super().__init__(embed_dim, context_dim, projector=projector,
                             n_queries=n_queries, **kw)

    if not kind:
        builder_mod.AttentionalPoolProjector = _Bound
        print(f"[flare-patch] AttentionalPoolProjector -> pooling ENABLED, "
              f"n_queries={n_queries}", flush=True)
        return

    def _make(embed_dim, context_dim, projector=None, **kw):
        hidden = embed_dim_out(projector)
        m = build_spatial_projector(kind, hidden)
        print(f"[flare-patch] projector -> {kind} (hidden={hidden}, "
              f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params)", flush=True)
        return m

    builder_mod.AttentionalPoolProjector = _make
