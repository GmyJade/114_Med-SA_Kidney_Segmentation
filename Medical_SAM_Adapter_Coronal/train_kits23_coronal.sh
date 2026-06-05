#!/bin/bash

python3 train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name kits23_Med-SA_train_coronal_EPOCH_500 \
    -encoder vit_b \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -image_size 1024 \
    -b 2 \
    -dataset kits \
    -data_path "/home/user412771064/project/Medical_project/畢專/dataset/KiTS23_for_MSA" \
    -num_sample 4 \
    -vis 5 \
    -slice_plane coronal
