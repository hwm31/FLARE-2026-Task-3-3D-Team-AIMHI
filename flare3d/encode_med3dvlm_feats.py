"""Extract DCFormer visual features from the Med3DVLM baseline.

The point is a cheap swap test. Eighteen runs on our own pipeline all failed the same way,
and the diagnosis landed on the FEATURES rather than the plumbing: the frozen BTB3D
classifier reads AUC 0.6593 (shuffled control 0.4597) yet picks findings worse than a
constant answer -- chest top-1 precision 0.201 against the prior's 0.497. Across-case
ranking exists; within-case ranking does not, and the metric needs the second one.

Med3DVLM ships a completely different visual encoder (DCFormer, contrastively pretrained
against ClinicalBERT) on the same task. Running the SAME probe on its features answers the
question the full fine-tune would take a day to answer:

    within-case precision beats the prior  ->  the encoder was the ceiling; port the recipe
    within-case precision still below it   ->  the ceiling is the task, not this encoder

Only the vision tower is loaded -- the 7B language model is irrelevant to the probe and
would dominate the memory budget.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
from tqdm import tqdm

MED = os.environ.get("MED3DVLM_DIR")
if not MED:
    raise SystemExit("set MED3DVLM_DIR to a checkout of https://github.com/mirthAI/Med3DVLM "
                     "(we used commit cbbd633)")
sys.path.insert(0, MED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vol-dir", required=True, help="preprocessed (1,128,256,256) .npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default=None, help="json split; default = every .npy present")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--streams", choices=["high", "both"], default="high",
                    help="high = only DCFormer's final 32x768 stream (the original, "
                         "spatially collapsed); both = also its 256x384 stream, tiled to "
                         "768 and concatenated, giving 288 tokens with spatial structure. "
                         "Defaults to `high` on purpose: it is the format every existing "
                         "feature directory and checkpoint was built with, and silently "
                         "changing it produced a corpus mixing 32- and 288-token volumes "
                         "that the collator could not batch.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from src.model.encoder.builder import build_vision_tower

    cfg_path = os.path.join(a.ckpt_dir, "config.json")
    raw = json.load(open(cfg_path))

    class Cfg:  # the builder reads plain attributes, not a HF config object
        vision_tower = raw["vision_tower"]
        dim = raw["dim"]
        depth = raw["depth"]
        input_size = tuple(raw["input_size"])
        vision_select_layer = raw["vision_select_layer"]
        vision_select_feature = raw["vision_select_feature"]

    tower = build_vision_tower(Cfg()).eval()

    # the vision weights live inside the sharded VLM checkpoint under vision_tower.*
    from safetensors.torch import load_file
    sd = {}
    for shard in sorted(glob.glob(os.path.join(a.ckpt_dir, "*.safetensors"))):
        for k, v in load_file(shard).items():
            if "vision_tower" in k:
                sd[k.split("vision_tower.", 1)[1] if "vision_tower." in k else k] = v
    if not sd:
        raise SystemExit(
            "No vision_tower.* tensors found. If --ckpt-dir points at\n"
            "  pretrained/vision_encoder.safetensors, do not use it: that file holds the\n"
            "  contrastive weights from BEFORE the FLARE fine-tuning. Renaming its keys makes\n"
            "  it load with 426/426 matched, which looks like success, but the resulting\n"
            "  features differ by 86%% (measured) and the pipeline silently produces garbage.\n"
            "  The weights actually used live in the VLM shards model-0000*-of-00004.safetensors.")
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    kept = len(sd) - len(unexpected)
    print(f"vision weights: matched {kept}/{len(sd)}  missing {len(missing)}", flush=True)
    assert kept > 0, ("no vision_tower tensors matched -- the probe would measure a randomly "
                      "initialised encoder and look like a clean negative")
    tower = tower.to("cuda", torch.bfloat16)

    if a.cases:
        want = {r["image"].replace(".nii.gz", "").replace(".npy", "")
                for r in json.loads(open(a.cases).read())}
    else:
        want = None
    files = sorted(glob.glob(os.path.join(a.vol_dir, "*.npy")))
    if want is not None:
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} volumes -> {a.out_dir}", flush=True)

    for f in tqdm(files, ncols=80):
        stem = os.path.basename(f)[:-4]
        out = os.path.join(a.out_dir, f"{stem}.nii_embedded.npz")
        if os.path.exists(out):
            continue
        x = torch.from_numpy(np.load(f)).unsqueeze(0).to("cuda", torch.bfloat16)
        with torch.inference_mode():
            z = tower(x)
        if isinstance(z, (tuple, list)):
            # DCFormer returns TWO streams and Med3DVLM's own projector ("mixer") consumes
            # both: (1, 256, 384) low and (1, 32, 768) high. Taking only z[-1] -- which is
            # what the first version of this script did -- throws away the 8x
            # higher-resolution stream, and with it every trace of WHERE anything is. That
            # is also why the pooled tokens collapse: 32 vectors expanded to 512 queries is
            # mostly duplication (measured within-case cosine +0.9939).
            #
            # The two are made concatenable without introducing any new parameters at
            # encode time: 384 tiles exactly twice into 768, so the low stream is repeated
            # along the channel axis rather than zero-padded or linearly projected. A
            # learned projection here would put untrained weights between the encoder and
            # every downstream measurement.
            if a.streams == "high":
                z = z[-1]
            else:
                lo, hi = z[0], z[-1]
                assert hi.shape[-1] % lo.shape[-1] == 0, \
                    f"cannot tile {lo.shape[-1]} into {hi.shape[-1]}"
                lo = lo.repeat(1, 1, hi.shape[-1] // lo.shape[-1])
                z = torch.cat([lo, hi], dim=1)          # (1, 256+32, 768)
        if z.ndim == 3:                       # (B, tokens, C) -> a 1-D "grid" the probe reads
            z = z.transpose(1, 2)[..., None, None]
        np.savez_compressed(out, arr=z.float().cpu().numpy().astype(np.float16))
    print("done", flush=True)


if __name__ == "__main__":
    main()
