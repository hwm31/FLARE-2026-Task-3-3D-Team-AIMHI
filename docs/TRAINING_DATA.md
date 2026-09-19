# Training data: format, provenance, and what is not included

The three adapters are trained on instruction files derived from the challenge's training split.
This page documents their format so they can be rebuilt, and states plainly which part of that
pipeline is not in this repository.

## Format

Every file is a JSON list of LLaVA-style records:

```json
{"id": "lvqa_amos_5756.nii.gz_1", "image": "amos_5756.nii.gz",
 "conversations": [{"from": "human", "value": "<image>\n..."}, {"from": "gpt", "value": "..."}]}
```

`image` is the case id with its `.nii.gz` suffix. The trainer looks up the visual features at
`<image_folder>/<stem>.nii_embedded.npz`, i.e. the `.nii.gz` suffix becomes `.nii` and `_embedded.npz`
is appended. The file holds one array `arr` of shape `(1, 768, 32, 1, 1)` in float16, as produced by
`flare3d/encode_med3dvlm_feats.py`. A missing feature file kills training within the first few steps,
so check every `image` in every split before launching.

Each adapter has a `<name>_train.json` and a `<name>_val.json`. The validation file is the held-out
partition used for checkpoint selection, not the challenge validation set.

## The three training splits

| Split | Adapter | Records in train | Composition |
|---|---|---|---|
| `stage2_mtstruct` | global | 5,822 | 2,911 `plcvqa_*` and 2,911 `struct_*` |
| `stage2_rag` | local | 15,493 | 12,582 `lvqa_*` and 2,911 `ragvqa_*` |
| `report_fillnatural` | report | 2,911 | 2,911 `report_*` |

All three cover the same 2,911 cases: 1,111 from CT-AMOS and 1,800 from CT-RATE.

### `plcvqa_*` (global adapter, main task)

One record per case. The prompt is built by `build_prompt` in
`flare3d/train_btb3d_3b/build_perlabelctx_split.py`: an instruction to answer `<condition>: yes` or
`<condition>: no` for each candidate, the case's own local questions as a bulleted list, and the
candidate list. Answer choices are stripped from the inserted questions. The answer is one
`<label>: yes|no` line per candidate. At inference the same prompt is produced by `perlabel_prompt` in
`infer_own_vqa.py` when `FLARE_GLOBAL_CTX=1`.

Two details of the training prompt differ from inference. The list of inserted questions is cut short
by a greedy regular expression in `local_questions`, which drops the third question from 98.1 % of
three-question chains, whereas inference inserts all of them. We measured the difference on the
challenge validation set and found none (0.3935 with the truncated list, 0.3938 with the full one), so
the released inference keeps the full list.

### `struct_*` (global adapter, auxiliary task)

One record per case: the model is asked, with no question list, whether each anatomical region is
`normal` or `abnormal`, one line per region.

```
You are a radiologist reporting a CT scan. For each anatomical region below, answer whether it is
normal or abnormal. Answer with exactly one line per region in the form '<region>: normal' or
'<region>: abnormal'.
Regions: liver; biliary system; spleen; pancreas; kidneys; endocrine system; lymphatic system;
gastrointestinal tract; abdominal cavity and peritoneum; blood vessels; musculoskeletal system;
lungs and pleura; respiratory tract; heart; mediastinum; esophagus; breast tissue; diaphragm
```

**The rule that derived these labels from the free-text findings is not recoverable.** The script that
produced them was not kept. A negation-based heuristic reproduces only 55.9 % of the labels, less than
the 73.5 % obtained by predicting `abnormal` everywhere, and the labels visibly disagree with the text
in places (a region described as "appears normal" is labelled `abnormal`). We found that the auxiliary
task helps regardless (see the paper, Section 5.1), which suggests that the exact labelling matters
little, but this repository cannot regenerate the original file.

### `lvqa_*` (local adapter, main task)

One record per question chain, with 12,582 in total. The questions are numbered in chain order with the
root first; a question with answer choices carries `Choices: [...]` as a Python list literal; the answer
joins the chain's answers with `|`:

```
<image>
1. Is there a stone in the left kidney?
2. Are there exudative lesions around the kidneys? Choices: ['No exudative lesions', ..., 'None of the above']
```
```
Yes|Mild exudative lesions around both kidneys
```

Chains are ordered by `score_vqa.build_chains`.

**`None of the above` matters.** The training and validation files were built differently, and the
difference is deterministic. We counted it over the released files:

| | Follow-up questions with a choice list | Last choice is `None of the above` | Answer is `None of the above` |
|---|---|---|---|
| Training | 17,441 | 17,441 (100 %) | 8,508: every follow-up under a root answered No, and none of the 8,933 under a root answered Yes |
| Validation | 3,323 | 0 | 0 |

In training, a No root therefore always comes with the answer `None of the above`. In validation that
option does not exist, and the 1,714 follow-ups under a No root carry ordinary answers. A model that has
learned the training rule emits the option on validation, where it is never valid. That is the origin of
the out-of-list predictions repaired by `flare3d/fix_offlist_followup.py`, and 8,508 / 17,441 = 48.8 % is
the share of training answers that sit in the last position.

### `ragvqa_*` (local adapter, auxiliary task)

A per-label global question whose prompt is prefixed with `[Similar prior cases]` and the findings of
retrieved neighbours from the training set. It is auxiliary training data only: retrieval is not used
at inference, and neither the local outputs nor the shipped inference code touch it. The name `m3d_rag`
of the local checkpoint comes from this split's name; the local answers themselves involve no retrieval.
The builder for this task is not included.

### `report_*` (report adapter)

The prompt lists 18 regions and asks for one `<region>: <findings>` line each. The answer contains all
19 regions, with `No abnormality is identified in the {region}.` in place of any region the
radiologist did not mention. `build_report_split.py --fill-normal natural` produces this format; an
independent reimplementation of it matched all 2,911 stored training records when we checked, and the
shipped script's output has the same shape (19 answer lines, natural-sentence fill).

## What is not included

| Missing | Effect |
|---|---|
| `split_manifest.json`: the case ids of the 2,911 / 324 partition (seed 42, validation fraction 0.1) | The exact development and held-out sets cannot be regenerated. |
| The builders that produced the base instruction file (`stage2_*.json`), the retrieval split, the per-label split, and the `struct_*` labels | Only `build_perlabelctx_split.py` and `build_report_split.py` are shipped, and the first of them reads a base file that is not. |
| The trained splits themselves | Not redistributable in any case: they contain challenge data under CC BY-NC 4.0. |

So the training recipe can be re-run from the commands in the README, but only on a split you rebuild
yourself from the formats above. The results in the paper were obtained with the original split.
Everything downstream of training (inference, post-processing, scoring) can be exercised without it,
provided you have adapters.
