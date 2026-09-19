"""Read the training configuration off a reference checkpoint, so the environment
variables are never written from memory.

We got this wrong twice while trying to reproduce the m3d_rag lineage: first by passing
PROJ=direct when the original uses an AttentionalPooler, then -- after fixing that -- by
passing N_QUERIES=512 when the original uses 32. Both times the value was assumed rather
than checked, and opening the checkpoint would have shown the error immediately.

  no attn_pool.query          -> PROJ=direct
  attn_pool.query of (N, C)   -> PROJ unset (AttentionalPooler), N_QUERIES=N
  mm_hidden_size                 -> MM_HIDDEN (768=DCFormer, 72=BTB3D-8)
"""
from __future__ import annotations
import json, os, sys
import torch

FL = os.environ.get("FLARE_ROOT", os.path.dirname(os.path.abspath(__file__)))


def config_of(run: str) -> dict:
    """`run` is a checkpoint directory, or a run name under <root>/train_btb3d_3b/ckpt/."""
    root = run if os.path.isdir(run) else f"{FL}/train_btb3d_3b/ckpt/{run}"
    nl = f"{root}/non_lora_trainables.bin"
    if not os.path.exists(nl):
        cs = sorted((d for d in os.listdir(root) if d.startswith("checkpoint-")),
                    key=lambda d: int(d.split("-")[1]))
        nl = f"{root}/{cs[-1]}/non_lora_trainables.bin"
    sd = torch.load(nl, map_location="cpu")
    q = [v for k, v in sd.items() if k.endswith("attn_pool.query")]
    cfg = json.load(open(f"{root}/config.json"))
    return {"MM_HIDDEN": cfg["mm_hidden_size"],
            "N_QUERIES": q[0].shape[0] if q else 512,
            "PROJ": "" if q else "direct"}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        c = config_of(run)
        print(f"{os.path.basename(os.path.normpath(run))}: " + " ".join(f"{k}={v}" for k, v in c.items()))
