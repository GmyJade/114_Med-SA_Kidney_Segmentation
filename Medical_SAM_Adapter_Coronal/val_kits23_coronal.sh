#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 val.py \
    -net sam \
    -mod sam_adpt \
    -exp_name val_kits23_coronal_full_slices \
    -encoder vit_b \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -weights ./logs/kits23_Med-SA_train_coronal_EPOCH_500_2026_04_30_18_19_25/Model/best_dice_checkpoint.pth \
    -image_size 1024 \
    -b 2 \
    -dataset kits \
    -data_path "/home/user412771064/project/Medical_project/畢專/dataset/KiTS23_for_MSA" \
    -num_sample 4 \
    -vis 1 \
    -slice_plane coronal
