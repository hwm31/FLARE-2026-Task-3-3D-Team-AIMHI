#!/bin/bash
# Report-generation LoRA. Trains on our-train only and evaluates on our held-out partition every
# 50 steps, so plateau/overfit is visible in real time instead of guessed from train loss. The
# challenge validation set stays untouched.
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# DeepSpeed needs a CUDA toolkit (nvcc). We used CUDA 13.0.88 with a PyTorch 2.4.1+cu121 build.
export CUDA_HOME="${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit that provides nvcc}"
# MM_HIDDEN and N_QUERIES have no default on purpose. They identify the checkpoint lineage
# (768 = DCFormer, 72 = the older BTB3D-8 encoder) and the pooler width, and a wrong value
# trains a different model without any error. Read them off a reference checkpoint with
# `python flare3d/ref_config.py <ckpt_dir>` -- see configs/adapters.json.
export FLARE_MM_HIDDEN="${MM_HIDDEN:?set MM_HIDDEN (768 for DCFormer features)}"
export FLARE_N_QUERIES="${N_QUERIES:?set N_QUERIES (32 local, 512 global/report)}"
# PROJ empty = AttentionalPooler, so every earlier report run reproduces byte-identically
export FLARE_PROJ=${PROJ:-}
# Root of the code tree (the directory that contains train_btb3d_3b/); override with FL=...
FL="${FL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$FL/train_btb3d_3b"

"${DEEPSPEED:-deepspeed}" --include localhost:${GPU:-0} --master_port $((29500 + ${GPU:-0} * 10 + 2)) \
  $FL/train_btb3d_3b/launch_train.py \
  --deepspeed "${BTB3D_DIR:?set BTB3D_DIR to a checkout of https://github.com/ibrahimethemhamamci/BTB3D}/report-generation/scripts/zero2.json" \
  --version llama3 \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type attn_pool+mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --report_to none \
  --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --model_name_or_path "${LLM_PATH:?set LLM_PATH to a local copy of meta-llama/Llama-3.2-3B-Instruct}" \
  --data_path "$FL/train_btb3d_3b/data_clean/${DATA:-report}_train.json" \
  --eval_data_path "$FL/train_btb3d_3b/data_clean/${DATA:-report}_val.json" \
  --image_folder "$FL/preproc/${IMG:-btb3d_enc_train}/" \
  $([ -z "${NOSTAGE1:-}" ] && echo "--pretrain_mm_mlp_adapter $FL/train_btb3d_3b/ckpt/${STAGE1:-clean_stage1}/mm_projector.bin") \
  --output_dir "$FL/train_btb3d_3b/ckpt/${RUN:-clean_report_lora}" \
  --num_train_epochs ${EPOCHS:-8} \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --evaluation_strategy steps --eval_steps 50 --per_device_eval_batch_size 4 \
  --save_strategy steps --save_steps 50 --save_total_limit 3 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --logging_steps 20
