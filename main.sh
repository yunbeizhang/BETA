#!/usr/bin/env bash
# -------------------------------------------------------------------
# BETA on ImageNet-C. Frozen black-box target: ViT-B/16. Local
# steering model: ViT-S/16. Adjust DATA_DIR to point to the root that
# contains `ImageNet/` and `ImageNet-C/`.
#
# Usage:   bash main.sh
# -------------------------------------------------------------------
set -e

export WANDB_SILENT="true"

: "${DATA_DIR:=/data}"

model=vitb16
local=vits16
pad_size=16
margin_e0=0.9
alpha=0.4
bvr_lr=0.01
norm_lr=0.00002
kl_weight=50
seed=2020
steps=1
corruption=all          # 'all' = all 15 ImageNet-C corruptions

python3 main.py \
    --algorithm beta \
    --model        ${model} \
    --local_helper ${local} \
    --pad_size     ${pad_size} \
    --alpha        ${alpha} \
    --margin_e0    ${margin_e0} \
    --bvr_lr       ${bvr_lr} \
    --norm_lr      ${norm_lr} \
    --kl_weight    ${kl_weight} \
    --steps        ${steps} \
    --seed         ${seed} \
    --batch_size   64 \
    --workers      5 \
    --corruption   ${corruption} \
    --data            ${DATA_DIR}/ImageNet \
    --data_corruption ${DATA_DIR}/ImageNet-C \
    --output ./outputs/${model}/beta \
    --tag _pad${pad_size}_lr${bvr_lr}_alpha${alpha}_seed${seed}_e${margin_e0}_nlr${norm_lr}_klw${kl_weight}
