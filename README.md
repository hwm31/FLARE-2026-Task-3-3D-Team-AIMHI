# FLARE 2026 Task 3 (3D CT) — Team AIMHI

Code for our submission to the FLARE 2026 3D CT track: region-wise report generation and two visual
question answering tasks (global and local) from the same CT volume.

One frozen DCFormer vision encoder and one frozen Llama-3.2-3B backbone serve all three tasks. Each task
adds its own LoRA adapter and projector; the volume is encoded once per case and the features are shared.

**Results on the 681 validation cases**, with the official Med3DVLM baseline re-measured under the same
preprocessing and scoring:

| Model | Global VQA | Local VQA | Report (GREEN) |
|---|---|---|---|
| Med3DVLM (baseline) | 0.2457 | 0.4231 | 0.4245 |
| **Ours** | **0.3938** | **0.5530** | **0.7883** |

See [Results](#7-results) for the ablations and for what these numbers do and do not show.

## Contents

1. [Environments and requirements](#1-environments-and-requirements)
2. [Dataset](#2-dataset)
3. [Preprocessing](#3-preprocessing)
4. [Training](#4-training)
5. [Inference](#5-inference)
6. [Evaluation](#6-evaluation)
7. [Results](#7-results)
8. [Reproducibility notes and known gaps](#8-reproducibility-notes-and-known-gaps)
9. [Contributing](#9-contributing)
10. [Acknowledgements](#10-acknowledgements)

```
configs/                 rules fitted on our held-out split, and the per-adapter settings
docs/TRAINING_DATA.md    format of the training files, and which parts of their pipeline are missing
environment/             exact package snapshots of the three environments
flare3d/                 preprocessing, feature extraction, post-processing, scoring
flare3d/train_btb3d_3b/  training, inference, and the two split builders
THIRD_PARTY_NOTICES.md   upstream licences and provenance
```

## 1. Environments and requirements

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| CPU | 2 x Intel Xeon Platinum 8462Y+ (128 threads) |
| RAM | 2 TB installed. Peak use was not measured. |
| GPU | 1 x NVIDIA H200 (141 GB), driver 580.126.16. Stage-2 training used up to about 65 GB. |
| CUDA | The PyTorch builds target CUDA 12.1 (model and GREEN environments) and 12.6 (data environment). Training needs `CUDA_HOME` to point at a toolkit with `nvcc` for DeepSpeed; we used 13.0.88. **Training only**: inference does not import DeepSpeed and ran without `CUDA_HOME`. |
| Python | 3.10.0 (model), 3.12.13 (data), 3.11.15 (GREEN) |

The three environments were built independently during the project and their PyTorch and Transformers
versions differ. We did not try to unify them, so this repository asks for three:

| Environment | Used for | Install | Exact snapshot |
|---|---|---|---|
| model | training, inference | `pip install -r requirements.txt` | `environment/model-py310.freeze.txt` |
| data | preprocessing, feature extraction, post-processing, VQA scoring | `pip install -r requirements-data.txt` | `environment/data-py312.freeze.txt` |
| GREEN | `flare3d/score_green.py` | `pip install -r requirements-green.txt` | `environment/green-py311.freeze.txt` |

The requirement files list direct dependencies; the snapshots list everything.
`requirements-green.txt` adds `typing_extensions`, which our own GREEN environment lacked when we
re-ran it (PyTorch imports it), so the scorer failed until it was supplied.

**Two external repositories are required**, at the commits we used:

```bash
git clone https://github.com/ibrahimethemhamamci/BTB3D && git -C BTB3D checkout 0eeb6e6
pip install -e BTB3D/report-generation          # model environment; provides the `llava` package
git clone https://github.com/mirthAI/Med3DVLM   && git -C Med3DVLM checkout cbbd633
export BTB3D_DIR=$PWD/BTB3D MED3DVLM_DIR=$PWD/Med3DVLM
```

BTB3D declares no licence; see `THIRD_PARTY_NOTICES.md`. We use its LLaVA training stack unmodified and
patch it in memory (`launch_train.py`, `infer_own_vqa.py`), and Med3DVLM for its vision-tower builder.

**Model weights**

| Component | Source | Notes |
|---|---|---|
| LLM | `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` | Gated. Download to a local directory and set `LLM_PATH`. |
| Vision encoder | `MagicXin/Med3DVLM-Qwen-2.5-7B-FLARE2025`, revision `8f208954945b701477bd44ee608f2c1c0be6dbc3` | Only the vision tower is used. |
| GREEN judge | `StanfordAIMI/GREEN-RadLlama2-7b` | Downloaded on first use. |

The DCFormer vision tower sits entirely in `model-00003-of-00004.safetensors` of the encoder repository.
Put that shard and `config.json` in one directory and pass it as `--ckpt-dir`. The other three shards are
the 7B language model and are not needed. We checked that this gives features identical to loading all four
(maximum difference 0.000).

> **Do not use `pretrained/vision_encoder.safetensors`** from that repository. Its keys can be made to load,
> and all 426 tensors match, but the file holds the contrastive weights from before the FLARE fine-tuning and
> the resulting features differ from the correct ones by 86 %. `encode_med3dvlm_feats.py` refuses it.

The scripts set `HF_HUB_OFFLINE=1`, so download the models first.

**Paths.** The code reads its locations from environment variables rather than from fixed paths:

| Variable | Meaning | Needed by |
|---|---|---|
| `MED3DVLM_DIR` | Med3DVLM checkout | `encode_med3dvlm_feats.py` |
| `BTB3D_DIR` | BTB3D checkout | training (for `zero2.json`) |
| `LLM_PATH` | local Llama-3.2-3B-Instruct directory | training, and inference when `--model-base` is omitted |
| `FLARE_TRAIN_DIR` | the dataset's `train/` directory | `fix_offlist_followup.py` without `--prior`, `build_report_split.py` |
| `CUDA_HOME` | CUDA toolkit with `nvcc` | training |
| `DEEPSPEED` | DeepSpeed launcher, if not on `PATH` | training |

## 2. Dataset

The FLARE 2026 Task 3 dataset is on the Hugging Face Hub at
https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-3D. Its README gives:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="FLARE-MedFM/FLARE-Task5-MLLM-3D", repo_type="dataset",
                  local_dir="./FLARE26-MLLM-3D", local_dir_use_symlinks=False, resume_download=True)
```

```
FLARE26-MLLM-3D/
├── train/
│   ├── CT-AMOS-1290/          images, source: https://era-ai-biomed.github.io/amos/
│   ├── CT-AMOS-Tr.json
│   ├── CT-RATE-2000/          images, source: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
│   └── CT-RATE-Tr.json
└── validation/
    ├── images/
    └── val.json               the test set is not released
```

The dataset is CC BY-NC 4.0. We use 3,236 training volumes (1,236 CT-AMOS, 2,000 CT-RATE) and the 681
validation volumes (387 abdominal, 294 chest). **We add no external labelled data.** Initial weights are the
public checkpoints listed above.

`CT-RATE-Tr.json` is spelled `CT-RATE-Tr .json` (with a space) in our copy. The code looks for both.

**Internal split.** We hold the challenge validation set out entirely during development. To pick
checkpoints and fit the post-hoc rules we split the training set into 2,911 development cases and 324
held-out cases. That split and the scripts that built the training files are **not included**; see
[section 8](#8-reproducibility-notes-and-known-gaps) and `docs/TRAINING_DATA.md`.

## 3. Preprocessing

The clipping window follows the challenge baseline (`FLARE25-MLLM3D/Data/process/process_ct.py`): CT-AMOS
is clipped to [-160, 240] HU and CT-RATE to [-1350, 150] HU, min-max normalised to [0, 1], and resampled
with anti-aliasing. We change only the target size, from 32x256x256 to 128x256x256, to match the encoder.

```bash
# data environment
python flare3d/preprocess_val.py \
    --json_in  FLARE26-MLLM-3D/validation/val.json \
    --nifti_dir FLARE26-MLLM-3D/validation/images \
    --out_base work/vol --workers 16
```

This writes `work/vol/med3dvlm/<case>.npy` (128x256x256), `work/vol/lamed/<case>.npy` (unused here), and
`work/vol/cases_processed.json`, the case list the later steps take as `--val-json`. The case ids in that
file end in `.npy`; the scripts accept either suffix.

Then extract the DCFormer features once. All three adapters read the same files:

```bash
python flare3d/encode_med3dvlm_feats.py \
    --ckpt-dir <dir with config.json and model-00003-of-00004.safetensors> \
    --vol-dir  work/vol/med3dvlm --out-dir work/feat
```

Each case gives `work/feat/<case>.nii_embedded.npz` holding 32 tokens of 768 channels. Cost per case:
0.17 s for preprocessing (16 workers) and 0.04 s for encoding.

## 4. Training

The three adapters differ in projector and in the number of visual queries, and a wrong value trains or
loads a differently shaped model. **`MM_HIDDEN` and `N_QUERIES` have no default** in the training scripts
for that reason. `configs/adapters.json` lists the values; you can also read them off any checkpoint:

```bash
python flare3d/ref_config.py <checkpoint dir> ...
```

| Adapter | Split | `MM_HIDDEN` | `N_QUERIES` | `PROJ` | Epochs | Wall-clock time, 1 x H200 |
|---|---|---|---|---|---|---|
| Global VQA | `stage2_mtstruct` | 768 | 512 | `direct` | 3 | 1 h 12 min |
| Local VQA | `stage2_rag` | 768 | **32** | *(empty)* | 3 | 2 h 05 min |
| Report | `report_fillnatural` | 768 | 512 | *(empty)* | 8 | 1 h 34 min |

The times are wall-clock from our logs, and the GPU was sometimes shared with other jobs. An empty `PROJ`
selects the attention-pool projector; `attnpool` is **not** a valid value.

Expected layout under `flare3d/`: split files in `train_btb3d_3b/data_clean/<DATA>_{train,val}.json`,
feature directories in `preproc/<IMG>/`, and checkpoints written to `train_btb3d_3b/ckpt/<RUN>/`.

```bash
export CUDA_HOME=<CUDA toolkit> BTB3D_DIR=<BTB3D checkout> LLM_PATH=<Llama-3.2-3B-Instruct directory>

# global VQA
env MM_HIDDEN=768 N_QUERIES=512 PROJ=direct DATA=stage2_mtstruct IMG=<feature dir> \
    RUN=mtstruct EPOCHS=3 NOSTAGE1=1 GPU=0 \
    bash flare3d/train_btb3d_3b/clean_stage2_loss2.sh

# local VQA
env MM_HIDDEN=768 N_QUERIES=32 DATA=stage2_rag IMG=<feature dir> \
    RUN=m3d_rag EPOCHS=3 NOSTAGE1=1 GPU=0 \
    bash flare3d/train_btb3d_3b/clean_stage2_loss2.sh

# report generation
env MM_HIDDEN=768 N_QUERIES=512 DATA=report_fillnatural IMG=<feature dir> \
    RUN=report_m3d_nos1 EPOCHS=8 NOSTAGE1=1 GPU=0 \
    bash flare3d/train_btb3d_3b/clean_report_m3d.sh
```

Shared settings: LoRA rank 64, alpha 128, dropout 0.05 on the query, key, value, output, gate, up and down
projections of all 28 layers (97.3 M parameters); learning rate 2e-4 with cosine decay and 3 % warmup;
per-device batch 4 with gradient accumulation 4 (effective 16); maximum length 2048; bf16 with DeepSpeed
ZeRO-2; checkpoint chosen by evaluation loss on the held-out split. The projector is trained too (11.8 M for
`direct`, 14.2 M and 14.6 M for the attention-pool variants). The vision encoder and the LLM stay frozen.
`SEED=<n>` sets the seed; without it the Hugging Face default (42) applies. There is no projector-alignment
pretraining stage (`NOSTAGE1=1`).

`--vision_tower openai/clip-vit-large-patch14-336` in the scripts is required by the LLaVA argument parser
and is kept for compatibility. The visual input is the pre-extracted DCFormer features.

To check the launch line without training, set `DEEPSPEED=echo`: the script then prints the full
command instead of running it. We compared that output with the commands recorded in our training logs for
one global-lineage run and one report run, and every token matched. The local settings were not compared
against a log.

**Pre-trained adapters** (209, 213 and 214 MB) are **not published yet**, so there is no download link.
Publishing them would carry the Llama 3.2 licence conditions and the non-commercial dataset terms.

## 5. Inference

Adapters are loaded from a checkpoint directory that holds `adapter_model.safetensors`,
`non_lora_trainables.bin`, `adapter_config.json` and `config.json`. `<LLM>` is the local
Llama-3.2-3B-Instruct directory. Run all of this in the model environment; `work/` is as in section 3.

```bash
# global VQA: one teacher-forced pass reads P(yes) for every candidate label, then thresholds them
env FLARE_MM_HIDDEN=768 FLARE_N_QUERIES=512 FLARE_PROJ=direct FLARE_GLOBAL_CTX=1 \
python flare3d/train_btb3d_3b/diag_perlabel_prob.py \
    --model-base <LLM> --checkpoint <global ckpt> \
    --val-json work/vol/cases_processed.json --enc-dir work/feat \
    --n-per-domain 999 --pred-csv work/global.csv --rule "chest:thr0.2,abd:thr0.2"

# local VQA: generate the chains, then read the root probability of each chain
env FLARE_MM_HIDDEN=768 FLARE_N_QUERIES=32 FLARE_PROJ= \
python flare3d/train_btb3d_3b/infer_own_vqa.py \
    --model-base <LLM> --checkpoint <local ckpt> \
    --val-json work/vol/cases_processed.json --enc-dir work/feat \
    --pred-csv work/local_gen.csv --global-format perlabel --local-format plain --image-pos front

env FLARE_MM_HIDDEN=768 FLARE_N_QUERIES=32 FLARE_PROJ= \
python flare3d/train_btb3d_3b/diag_local_root_prob.py \
    --model-base <LLM> --checkpoint <local ckpt> \
    --val-json work/vol/cases_processed.json --enc-dir work/feat --out-json work/root.json

# report generation
env FLARE_MM_HIDDEN=768 FLARE_N_QUERIES=512 FLARE_PROJ= \
python flare3d/train_btb3d_3b/infer_own_report.py \
    --model-base <LLM> --checkpoint <report ckpt> \
    --val-json work/vol/cases_processed.json --enc-dir work/feat \
    --report-csv work/report.csv --max-new-tokens 1400
```

Three things are easy to get wrong here:

- `--n-per-domain` is how many cases to process per domain. **`0` means zero, not unlimited**, and yields an
  empty file. Pass a number larger than the split.
- The label-selection rule is in `configs/global_rule.json`. The code default, `chest:top5`, was chosen for an
  earlier model and scores 0.3530 with this one, against 0.3938 for `chest:thr0.2`.
- `FLARE_GLOBAL_CTX=1` is the prompt format the global adapter was trained with. With it off, the prompt
  no longer matches training.

**Post-processing.** The local score is not the generation's: it is the end of a three-step chain,
0.5063 (generation), 0.5229 (root calibration), 0.5530 (out-of-list correction).

```bash
python flare3d/apply_postproc.py --pred-csv work/local_gen.csv --root-json work/root.json \
    --thresholds configs/root_thresholds.json --val-json work/vol/cases_processed.json \
    --out-csv work/local_calib.csv

python flare3d/fix_offlist_followup.py --pred-csv work/local_calib.csv \
    --val-json work/vol/cases_processed.json --prior configs/offlist_prior.json \
    --out-csv work/local_final.csv

python flare3d/assemble_predictions.py --global-csv work/global.csv \
    --local-csv work/local_final.csv --out-csv work/predictions.csv
```

`configs/` holds the rules used for the results, all fitted on our held-out split and applied unchanged to
validation: `root_thresholds.json` (the P(yes) threshold per chain length: 0.51, 0.53, 0.62),
`offlist_prior.json` (the replacement answer position by choice-list size), and `global_rule.json`.
`fix_offlist_followup.py` can instead compute the prior from the training files
(`FLARE_TRAIN_DIR=... `, no `--prior`); the two routes give byte-identical output.

**Docker.** The challenge container was built separately and is not part of this repository. No image is
published.

**Runtime.** Measured on an idle H200 by timing runs of 10 and 40 cases and taking the slope, which
separates model loading from the per-case cost:

| Step | s per case |
|---|---|
| preprocessing + encoding | 0.21 |
| global VQA | 0.19 |
| local VQA | 7.03 |
| report | 4.83 |
| **total** | **12.26** |

The Med3DVLM baseline costs 9.65 s per case (4.06 VQA + 5.59 report), so ours is 1.27 times the baseline.
**The baseline figure is not measured under the same conditions**: it comes from the timing of our earlier
full runs, when the device may have been shared with other jobs, and we did not re-measure it on an idle
device. The ratio is therefore approximate. Local VQA is 57 % of our total because a case carries 4.2 chains
on average (2,891 chains over 681 cases).

## 6. Evaluation

```bash
# data environment
python flare3d/score_vqa.py --pred_csv work/predictions.csv \
    --val_json FLARE26-MLLM-3D/validation/val.json --out_json work/vqa.json

# GREEN environment
python flare3d/score_green.py --report_csv work/report.csv --out_json work/green.json
```

`score_vqa.py` and `score_green.py` follow the challenge's reference scorers (see
`THIRD_PARTY_NOTICES.md`). The region-wise use of GREEN is the challenge's protocol, not part of the GREEN
metric: the evaluation driver parses each report into per-region text and applies GREEN to each region.

**Empty predictions must be written as a single space, not an empty string.** An empty CSV field is read back
as `NaN`, whose string form counts as one predicted label that does not exist. Our global adapter predicts
nothing for 16 of the 681 validation cases, 12 of them with an empty reference, and writing a space is worth
0.0176 global accuracy (0.3762 to 0.3938). `assemble_predictions.py` does this. We applied the same
correction to the baseline's predictions (34 empty predictions in its output file), so the comparison is
unaffected.

Do not modify `score_vqa.py` to fix this: reading with `keep_default_na=False` breaks the `question_id`
parsing. Fix the writer instead.

## 7. Results

Validation set, 681 cases. GREEN is scored by the challenge's region-wise protocol.

| Model | Global VQA | Local VQA | Report (GREEN) |
|---|---|---|---|
| Med3DVLM (baseline) | 0.2457 | 0.4231 | 0.4245 |
| **Ours** | **0.3938** | **0.5530** | **0.7883** |

The global figure is the submitted checkpoint. Across six training seeds the same configuration averages
0.3877 +/- 0.0041, so the submitted value is within one standard deviation of the family mean. Single-run
maxima are not reproducible quantities, and several design decisions below involve differences smaller than
this spread.

**Global VQA**, each row a family mean over *n* seeds:

| Configuration | *n* | Score |
|---|---|---|
| Per-label prompt, attention-pool projector | 5 | 0.2659 +/- 0.0038 |
| Per-label prompt, identity projector | 2 | 0.2767 +/- 0.0090 |
| + local questions inserted into the prompt | 4 | 0.3593 +/- 0.0069 |
| + region-wise normal/abnormal supervision | 6 | 0.3877 +/- 0.0041 |

Predicting the most frequent training labels for every case, without reading the image or the question,
scores 0.2647. Zeroing the visual tokens of the final model costs 0.040, so the image accounts for about
10 % of the score.

**Local VQA**, one generation model (root accuracy, conditional chain accuracy, score):

| Stage | Root | Conditional | Local |
|---|---|---|---|
| Generation | 0.6098 | 0.8303 | 0.5063 |
| + root calibration | 0.6264 | 0.8348 | 0.5229 |
| + out-of-list correction | 0.6264 | 0.8828 | 0.5530 |

**Report.** Coverage-complete targets raise GREEN from 0.4245 to 0.7883, but most of that is coverage. The
baseline mentions 82 % of the reference regions (averaged over the 16 regions with at least 20 reference
occurrences) and ours 100 %. On the 4,389 region-case pairs both models write about, GREEN is 0.699 against
0.748 on regions the reference calls normal and 0.323 against 0.364 on regions it calls abnormal. Both models
are weak where the reference reports an abnormality. The gain is largely one of answering the question the
protocol asks, and only partly one of describing findings better.

**Checking this release.** After preparing it we ran the released code end to end, from NIfTI to scores, on
five validation cases with the shipped weights. For every prediction we had recorded from the full runs
(three global label lists, five local chains after post-processing, six report regions), the new output was
identical. On those five cases GREEN was 0.7951 and VQA was 0.474 (global) and 0.735 (local), which checks
the machinery and is not a benchmark. We did not re-run the 681-case evaluation with this exact snapshot.

## 8. Reproducibility notes and known gaps

- **The training files cannot be regenerated from this repository.** The case-id manifest of the
  2,911 / 324 split and the builders of the base instruction file, the retrieval split, the per-label split and
  the region-wise labels are not included. Only two builders are, and one of them reads a file that is not.
  `docs/TRAINING_DATA.md` gives the formats. The training commands are therefore correct but need a split you
  rebuild yourself.
- **The rule behind the region-wise labels is unknown.** The script was not kept and a negation heuristic
  agrees with only 55.9 % of the labels. The auxiliary task helps regardless.
- **`attn_pool_projector.py` was reconstructed** from a record of the file made during the project, after we
  found it missing from an earlier packaging of this release. It matches the released checkpoints (14 tensors
  each, identical names and shapes and parameter counts) and reproduced every prediction we compared. It
  keeps disabled experimental hooks that the final adapters do not use.
- **Predictions were identical on the cases we compared, not proved bitwise-reproducible** across hardware.
- **Three environments** are needed; see section 1.
- **Licence.** The repository's `LICENSE` (MIT) covers our own code. It does not cover third-party material: `flare3d/train_btb3d_3b/attn_pool_projector.py` contains a class copied from BTB3D, whose terms are unverified, and the model weights and datasets carry their own licences. See `THIRD_PARTY_NOTICES.md`.

## 9. Contributing

Bug reports and questions are welcome as issues. For a bug, include the command, the environment snapshot you
used, and the first error. Please open an issue before starting a large change.

## 10. Acknowledgements

We thank the FLARE 2026 organisers for the challenge and the dataset, and the authors of the resources this
work builds on:

- The challenge dataset, built from CT-AMOS and CT-RATE:
  - Ji, Yuanfeng, et al. "AMOS: A large-scale abdominal multi-organ benchmark for versatile medical image
    segmentation." *Advances in Neural Information Processing Systems* 35 (2022): 36722-36732.
  - Hamamci, Ibrahim Ethem, et al. "Developing generalist foundation models from a multimodal dataset for 3D
    computed tomography." arXiv:2403.17834 (2024).
- Med3DVLM (https://github.com/mirthAI/Med3DVLM) for the baseline, the DCFormer encoder, and the scoring and
  preprocessing code we follow.
- The FLARE25-MLLM3D baseline (https://github.com/medfm-flare/FLARE25-MLLM3D).
- BTB3D (https://github.com/ibrahimethemhamamci/BTB3D) for the LLaVA-based training stack.
- Meta for Llama 3.2.
- Ostmeier and Delbrouck for the GREEN metric (`green-score`) and the GREEN judge model.
