# Third-party notices

This repository builds on the work listed below. Nothing here replaces the upstream terms; where a
licence could not be verified, that is stated rather than assumed.

## 1. Code derived from MIT-licensed challenge baselines

Three files follow the challenge baselines closely so that our numbers are comparable to theirs.
The statements below repeat what the files' own headers say.

| File | Derived from |
|---|---|
| `flare3d/score_vqa.py` | The scoring functions of Med3DVLM (`src/flare_infer/eval_vqa.py`) and FLARE25-MLLM3D (`eval_vqa.py`), which are identical in logic. Only the command-line interface and some diagnostic counters are ours. |
| `flare3d/score_green.py` | The region-wise procedure of `generate_green_score.py`, identical in the Med3DVLM and FLARE25-MLLM3D repositories. |
| `flare3d/preprocess_val.py` | Med3DVLM `flare_preprocess.py` and FLARE25-MLLM3D `Data/process/process_ct.py`, which differ only in the resize target. |

Both upstream repositories are distributed under the MIT licence. Their notices follow, copied
verbatim from the `LICENSE` file of each repository.

### Med3DVLM

Source: https://github.com/mirthAI/Med3DVLM (we used commit `cbbd633`)

```
MIT License

Copyright (c) 2025 mirth AI lab at UF

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### FLARE25-MLLM3D

Source: https://github.com/medfm-flare/FLARE25-MLLM3D (we used commit `53dfb6d`)

```
MIT License

Copyright (c) 2024 BAAI-DCAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Code adapted from BTB3D

| File | Relationship |
|---|---|
| `flare3d/train_btb3d_3b/attn_pool_projector.py` | The `AttentionalPooler` class is copied from BTB3D's `report-generation/llava/model/multimodal_projector/coca_attentional_pooler.py` (commit `0eeb6e6`); the rest of the file is ours. |

**Licence status: unverified.** BTB3D (https://github.com/ibrahimethemhamamci/BTB3D) declares no
licence: the GitHub licence endpoint returns 404 for it and the checkout we used contains no
`LICENSE` file. Absent a licence, redistribution is not granted by default. Anyone reusing
`attn_pool_projector.py` should confirm terms with the BTB3D authors first.

The remaining relationship to BTB3D is by import, not by copy. `launch_train.py` and
`infer_own_vqa.py` use BTB3D's `llava` package and its DeepSpeed configuration
(`report-generation/scripts/zero2.json`) at run time, and patch them in memory. We do not
redistribute either.

## 3. Used but not redistributed

| Component | Terms |
|---|---|
| Llama-3.2-3B-Instruct (`meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`) | Llama 3.2 Community License; access is gated on the Hugging Face Hub. Adapters trained on top of it are derivative works and carry its conditions. |
| DCFormer vision tower, taken from `MagicXin/Med3DVLM-Qwen-2.5-7B-FLARE2025` (revision `8f208954945b701477bd44ee608f2c1c0be6dbc3`) | See the model card. We did not verify the licence of this checkpoint. |
| FLARE 2026 Task 3 dataset | CC BY-NC 4.0 according to the dataset's own README (front matter `license: cc-by-nc-4.0`). Non-commercial use only. |
| CT-RATE and AMOS (the sources of the dataset) | See the source pages linked in the README. |
| `green-score` 0.0.12 (Ostmeier and Delbrouck) | MIT, according to the package metadata. |
| `StanfordAIMI/GREEN-RadLlama2-7b` (GREEN judge model) | See the model card. |

Trained adapter weights are not part of this repository. Publishing them would carry the Llama
3.2 conditions and the non-commercial dataset terms above.
