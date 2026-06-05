import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
# 引入 PIL 用於繪製 Times New Roman 字體
from PIL import Image, ImageDraw, ImageFont 
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
from conf import settings
from utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def _metadata_value(value, index, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        return value[index].item()
    if isinstance(value, np.ndarray):
        return value[index].item() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value

def _parse_case_id(name):
    base = os.path.basename(str(name)).replace('.nii.gz', '')
    if '_slice' in base:
        return base.split('_slice')[0]
    return base

def _parse_slice_idx(name, default=0):
    text = str(name)
    if '_slice' not in text:
        return default
    try:
        return int(text.rsplit('_slice', 1)[1])
    except ValueError:
        return default

def _area_mm2(area_pixels, pixel_area_mm2):
    if pixel_area_mm2 is None:
        return None
    return float(area_pixels) * float(pixel_area_mm2)

def _safe_filename(text):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(text))

def _sort_kits_top_records(records):
    return sorted(records, key=lambda r: (-r['pred_area_pixels'], r['slice_idx'], r['filename']))

def _update_kits_topk_visuals(topk_map, record, origin_img, pred_prob, true_mask, top_k=20):
    case_id = record['case_id']
    candidates = topk_map.setdefault(case_id, [])
    qualifies = len(candidates) < top_k
    if not qualifies:
        weakest = _sort_kits_top_records([c['record'] for c in candidates])[-1]
        qualifies = _sort_kits_top_records([record, weakest])[0] is record
    if not qualifies:
        return

    candidates.append({
        'record': record.copy(),
        'origin_img': origin_img.detach().cpu().clone(),
        'pred_prob': pred_prob.detach().cpu().clone(),
        'true_mask': true_mask.detach().cpu().clone(),
    })
    candidates.sort(key=lambda c: (-c['record']['pred_area_pixels'], c['record']['slice_idx'], c['record']['filename']))
    del candidates[top_k:]

def _write_kits_area_summary(records, output_dir, topk_visuals=None, sample_dir=None, top_k=20, epoch=None):
    if not records:
        return None, None, None, None

    os.makedirs(output_dir, exist_ok=True)
    per_slice_path = os.path.join(output_dir, 'kits_pred_mask_area_per_slice.csv')
    summary_path = os.path.join(output_dir, 'kits_pred_mask_area_summary.csv')
    topk_path = os.path.join(output_dir, f'kits_pred_mask_area_top{top_k}_per_case.csv')
    topk_case_dir = os.path.join(output_dir, f'kits_pred_mask_area_top{top_k}_by_case')

    records = sorted(records, key=lambda r: (r['case_id'], r['slice_idx']))
    with open(per_slice_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'case_id',
                'slice_idx',
                'raw_pred_area_pixels',
                'pred_area_pixels',
                'gt_area_pixels',
                'resized_pixel_area_mm2',
                'raw_pred_area_mm2',
                'pred_area_mm2',
                'gt_area_mm2',
                'dice',
                'iou',
                'filename',
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    case_map = {}
    for record in records:
        case_map.setdefault(record['case_id'], []).append(record)

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'case_id',
                'max_pred_slice_idx',
                'max_pred_area_pixels',
                'max_pred_area_mm2',
                'gt_area_pixels_at_max_pred',
                'gt_area_mm2_at_max_pred',
                'num_slices_evaluated',
            ],
        )
        writer.writeheader()
        for case_id, case_records in sorted(case_map.items()):
            max_record = max(case_records, key=lambda r: (r['pred_area_pixels'], -r['slice_idx']))
            writer.writerow({
                'case_id': case_id,
                'max_pred_slice_idx': max_record['slice_idx'],
                'max_pred_area_pixels': max_record['pred_area_pixels'],
                'max_pred_area_mm2': max_record['pred_area_mm2'],
                'gt_area_pixels_at_max_pred': max_record['gt_area_pixels'],
                'gt_area_mm2_at_max_pred': max_record['gt_area_mm2'],
                'num_slices_evaluated': len(case_records),
            })

    topk_fieldnames = [
        'case_id',
        'rank',
        'slice_idx',
        'raw_pred_area_pixels',
        'pred_area_pixels',
        'gt_area_pixels',
        'resized_pixel_area_mm2',
        'raw_pred_area_mm2',
        'pred_area_mm2',
        'gt_area_mm2',
        'dice',
        'iou',
        'filename',
        'visualization_path',
    ]
    topk_image_dir = os.path.join(sample_dir, f'kits_pred_mask_area_top{top_k}_by_case') if sample_dir else None
    visualization_paths = {}
    if topk_visuals and topk_image_dir:
        for case_id, candidates in sorted(topk_visuals.items()):
            case_image_dir = os.path.join(topk_image_dir, _safe_filename(case_id))
            candidates.sort(key=lambda c: (-c['record']['pred_area_pixels'], c['record']['slice_idx'], c['record']['filename']))
            for rank, candidate in enumerate(candidates[:top_k], start=1):
                record = candidate['record']
                image_name = (
                    f"rank{rank:02d}_slice{int(record['slice_idx']):04d}_"
                    f"area{int(record['pred_area_pixels'])}_epoch{epoch}.jpg"
                )
                visualization_paths[(record['case_id'], record['slice_idx'], record['filename'])] = os.path.join(
                    case_image_dir,
                    image_name,
                )

    topk_rows_by_case = {}
    with open(topk_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=topk_fieldnames)
        writer.writeheader()
        for case_id, case_records in sorted(case_map.items()):
            ranked_records = _sort_kits_top_records(case_records)[:top_k]
            topk_rows_by_case[case_id] = []
            for rank, record in enumerate(ranked_records, start=1):
                row = record.copy()
                row['rank'] = rank
                row['visualization_path'] = visualization_paths.get((record['case_id'], record['slice_idx'], record['filename']), '')
                topk_rows_by_case[case_id].append(row)
                writer.writerow({field: row.get(field) for field in topk_fieldnames})

    os.makedirs(topk_case_dir, exist_ok=True)
    for case_id, rows in topk_rows_by_case.items():
        case_csv_path = os.path.join(topk_case_dir, f'{_safe_filename(case_id)}_top{top_k}.csv')
        with open(case_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=topk_fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in topk_fieldnames} for row in rows)

    if topk_visuals and sample_dir:
        os.makedirs(topk_image_dir, exist_ok=True)
        for case_id, candidates in sorted(topk_visuals.items()):
            case_image_dir = os.path.join(topk_image_dir, _safe_filename(case_id))
            os.makedirs(case_image_dir, exist_ok=True)
            candidates.sort(key=lambda c: (-c['record']['pred_area_pixels'], c['record']['slice_idx'], c['record']['filename']))
            for rank, candidate in enumerate(candidates[:top_k], start=1):
                record = candidate['record']
                image_name = (
                    f"rank{rank:02d}_slice{int(record['slice_idx']):04d}_"
                    f"area{int(record['pred_area_pixels'])}_epoch{epoch}.jpg"
                )
                image_path = visualization_paths.get(
                    (record['case_id'], record['slice_idx'], record['filename']),
                    os.path.join(case_image_dir, image_name),
                )
                debug_dice_visualization(
                    candidate['origin_img'],
                    candidate['pred_prob'],
                    candidate['true_mask'],
                    image_path,
                    threshold=0.5,
                )

    return per_slice_path, summary_path, topk_path, topk_case_dir

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            masks = (masks > 0).float()
            
            imgs, pt, masks, generated_labels = generate_click_prompt(imgs, masks)
            point_labels = generated_labels
            
            name = pack['image_meta_dict']['filename_or_obj']

            # print("before thd branch imgs.shape =", imgs.shape)
            # print("before thd branch masks.shape =", masks.shape)
            # print("before thd branch pt.shape =", pt.shape)
            
            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                generated_labels = rearrange(generated_labels, 'b d -> (b d)')
                point_labels = generated_labels 
                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                imgs = imgs.repeat(1,3,1,1)
                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                showp = pt.float().clone()
                showp[..., 0] = showp[..., 0] * (args.out_size / mask_h)
                showp[..., 1] = showp[..., 1] * (args.out_size / mask_w)
                pt = pt.float()
                pt[..., 0] = pt[..., 0] * (args.image_size / mask_h)
                pt[..., 1] = pt[..., 1] * (args.image_size / mask_w)
            else:
                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                showp = pt.float().clone()
                showp[..., 0] = showp[..., 0] * (args.out_size / mask_h)
                showp[..., 1] = showp[..., 1] * (args.out_size / mask_w)
                pt = pt.float()
                pt[..., 0] = pt[..., 0] * (args.image_size / mask_h)
                pt[..., 1] = pt[..., 1] * (args.image_size / mask_w)

            mask_type = torch.float32
            ind += 1
            point_coords = pt[..., [1, 0]] 
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
            if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
            pt_model = (coords_torch, labels_torch)

            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n: value.requires_grad = False
                    else: value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(net.image_encoder, lora_r=4, target_rank=8, init_warmup=500, final_warmup=1500, mask_interval=10, total_step=3000, beta1=0.85, beta2=0.85)
            else:
                for n, value in net.image_encoder.named_parameters(): value.requires_grad = True

            origin_imgs = imgs.clone()        
            imgs = net.preprocess(imgs)        
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
                elif args.net == "efficient_sam":
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                    se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)
            
            if args.net == 'sam':
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1))
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
            elif args.net == "efficient_sam":
                se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, multimask_output=False)

            pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)
            loss = lossfunc(pred, masks)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            if vis and ind % vis == 0:
                
                import os

                namecat = 'Train'
                for na in name[:2]:
                    base = os.path.basename(na)
                    base = base.replace('.nii.gz', '')
                    namecat = namecat + base + '+'                
                
                show_labels = point_labels.clone()
                vis_image(origin_imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp, point_labels=show_labels)

            pbar.update()

    return epoch_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_slices_processed = 0
    noise_filtered_count = 0
    collect_kits_area = args.dataset == 'kits' and bool(getattr(args, 'full_slice_eval', False))
    kits_area_records = []
    kits_top20_visuals = {}
    
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            masksw = (masksw > 0).float()
            cur_bsz = imgsw.shape[0]
            
            # Force generate prompts
            imgsw, ptw, masksw, generated_labels_w = generate_click_prompt(imgsw, masksw)
            point_labels = generated_labels_w
            name = pack['image_meta_dict']['filename_or_obj']     
            
            buoy = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                batch_labels = generated_labels_w[:, buoy: buoy + evl_ch] if args.thd else point_labels
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    labels_torch = rearrange(batch_labels, 'b d -> (b d)')
                    mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                    imgs = imgs.repeat(1,3,1,1)
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                    showp = pt.float().clone()
                    showp[..., 0] = showp[..., 0] * (args.out_size / mask_h)
                    showp[..., 1] = showp[..., 1] * (args.out_size / mask_w)
                    pt = pt.float()
                    pt[..., 0] = pt[..., 0] * (args.image_size / mask_h)
                    pt[..., 1] = pt[..., 1] * (args.image_size / mask_w)
                else:
                    mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                    showp = pt.float().clone()
                    showp[..., 0] = showp[..., 0] * (args.out_size / mask_h)
                    showp[..., 1] = showp[..., 1] * (args.out_size / mask_w)
                    pt = pt.float()
                    pt[..., 0] = pt[..., 0] * (args.image_size / mask_h)
                    pt[..., 1] = pt[..., 1] * (args.image_size / mask_w)
                    current_point_labels = batch_labels
                
                ind += 1
                point_coords = pt[..., [1, 0]]
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                if not args.thd: labels_torch = torch.as_tensor(current_point_labels, dtype=torch.int, device=GPUdevice)
                if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
                if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
                pt_model = (coords_torch, labels_torch)

                mask_type = torch.float32
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                with torch.no_grad():
                    origin_imgs = imgs.clone()
                    imgs = net.preprocess(imgs)
                    imge= net.image_encoder(imgs)
                    
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1))
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
                    elif args.net == "efficient_sam":
                        se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, multimask_output=False)

                    pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)
                    total_loss += lossfunc(pred, masks) * cur_bsz

                    # Dice/IoU Calculation with Noise Filter
                    pred_prob = torch.sigmoid(pred)
                    pred_binary = (pred_prob > 0.5).float()
                    masks_binary = (masks > 0.5).float()
                    # 不再使用黑盒的 eval_seg()，改成手動計算每張切片的指標，方便之後加入 noise filter 與 slice-level debug

                    current_batch_size = pred_binary.shape[0]
                    total_slices_processed += current_batch_size 

                    if collect_kits_area:
                        meta = pack.get('image_meta_dict', {})
                        filenames = meta.get('filename_or_obj', name)
                        case_ids = meta.get('case_id')
                        slice_idxs = meta.get('slice_idx')
                        resized_pixel_area_mm2_values = meta.get('resized_pixel_area_mm2')

                    for i in range(current_batch_size):
                        p_i = pred_binary[i]
                        t_i = masks_binary[i]
                        raw_pred_area_pixels = int(p_i.sum().item())
                        
                        # Noise Filter (Threshold = 50)
                        if p_i.sum() < 50:
                            if p_i.sum() > 0: noise_filtered_count += 1
                            p_i = torch.zeros_like(p_i)

                        intersection = (p_i * t_i).sum()
                        total_area = p_i.sum() + t_i.sum()
                        union_area = total_area - intersection
                        
                        if total_area < 1: dice_score = 1.0
                        else: dice_score = float(((2.0 * intersection) / (total_area + 1e-6)).item())
                        
                        if union_area < 1: iou_score = 1.0
                        else: iou_score = float((intersection / (union_area + 1e-6)).item())
                            
                        total_dice += dice_score
                        total_iou += iou_score

                        if collect_kits_area:
                            filename = str(_metadata_value(filenames, i, ''))
                            case_id = _metadata_value(case_ids, i, None)
                            slice_idx = _metadata_value(slice_idxs, i, None)
                            resized_pixel_area_mm2 = _metadata_value(resized_pixel_area_mm2_values, i, None)
                            if case_id is None:
                                case_id = _parse_case_id(filename)
                            if slice_idx is None:
                                slice_idx = _parse_slice_idx(filename, default=i)

                            pred_area_pixels = int(p_i.sum().item())
                            gt_area_pixels = int(t_i.sum().item())
                            area_record = {
                                'case_id': str(case_id),
                                'slice_idx': int(slice_idx),
                                'raw_pred_area_pixels': raw_pred_area_pixels,
                                'pred_area_pixels': pred_area_pixels,
                                'gt_area_pixels': gt_area_pixels,
                                'resized_pixel_area_mm2': resized_pixel_area_mm2,
                                'raw_pred_area_mm2': _area_mm2(raw_pred_area_pixels, resized_pixel_area_mm2),
                                'pred_area_mm2': _area_mm2(pred_area_pixels, resized_pixel_area_mm2),
                                'gt_area_mm2': _area_mm2(gt_area_pixels, resized_pixel_area_mm2),
                                'dice': dice_score,
                                'iou': iou_score,
                                'filename': filename,
                            }
                            kits_area_records.append(area_record)
                            _update_kits_topk_visuals(
                                kits_top20_visuals,
                                area_record,
                                origin_imgs[i],
                                pred_prob[i],
                                masks[i],
                                top_k=20,
                            )

                    # Visualizations
                    if args.vis and ind % args.vis == 0:
                        
                        import os

                        namecat = 'Test'
                        for na in name[:2]:
                            base = os.path.basename(na)          # case_00422.nii.gz_slice84
                            base = base.replace('.nii.gz', '')   # case_00422_slice84
                            namecat = namecat + base + '+'

                        show_labels = labels_torch.clone()
                        vis_image(
                            origin_imgs,
                            pred,
                            masks,
                            os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                            reverse=False,
                            points=showp,
                            point_labels=show_labels
                        )

                        batch_n = min(len(name), origin_imgs.shape[0], pred_prob.shape[0], masks.shape[0])

                        for i in range(batch_n):
                            one_name = os.path.basename(name[i])          # case_00422.nii.gz_slice84
                            one_name = one_name.replace('.nii.gz', '')    # case_00422_slice84

                            debug_save_path = os.path.join(
                                args.path_helper['sample_path'],
                                f"{one_name}_dice_epoch{epoch}.jpg"
                            )

                            debug_dice_visualization(
                                origin_imgs[i],
                                pred_prob[i],
                                masks[i],
                                debug_save_path,
                                threshold=0.5
                            )
                        
            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    if total_slices_processed == 0: total_slices_processed = 1
    avg_loss = total_loss / total_slices_processed
    avg_iou = total_iou / total_slices_processed
    avg_dice = total_dice / total_slices_processed

    if collect_kits_area:
        per_slice_path, summary_path, top20_path, top20_case_dir = _write_kits_area_summary(
            kits_area_records,
            args.path_helper['log_path'],
            topk_visuals=kits_top20_visuals,
            sample_dir=args.path_helper.get('sample_path'),
            top_k=20,
            epoch=epoch,
        )
        if per_slice_path and summary_path:
            print(f"[KiTS area] per-slice mask area saved to: {per_slice_path}")
            print(f"[KiTS area] case max-slice summary saved to: {summary_path}")
            print(f"[KiTS area] case top-20 slice area saved to: {top20_path}")
            print(f"[KiTS area] per-case top-20 CSV files saved to: {top20_case_dir}")
            print(f"[KiTS area] per-case top-20 images saved to: {os.path.join(args.path_helper['sample_path'], 'kits_pred_mask_area_top20_by_case')}")

    return avg_loss, (avg_iou, avg_dice)

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1); label = label.transpose(0,1)
    coord = coord.unsqueeze(1); label = label.unsqueeze(1)
    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)
    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[:, :, : decoder_max_num_input_points, :]
        label = label[:, :, : decoder_max_num_input_points]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(rescaled_batched_points, (0, 0, 0, decoder_max_num_input_points - num_pts), value=-1.0)
        label = F.pad(label, (0, decoder_max_num_input_points - num_pts), value=-1.0)
    rescaled_batched_points = rescaled_batched_points.reshape(batch_size * max_num_queries, decoder_max_num_input_points, 2)
    label = label.reshape(batch_size * max_num_queries, decoder_max_num_input_points)
    return rescaled_batched_points,label

def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
    return torch.stack([
        torch.where(batched_points[..., 0] >= 0, batched_points[..., 0] * 1024 / input_w, -1.0),
        torch.where(batched_points[..., 1] >= 0, batched_points[..., 1] * 1024 / input_h, -1.0),
    ], dim=-1)

def debug_dice_visualization(origin_img, pred_prob, true_mask, save_path, threshold=0.5):
    """
    Overlay visualization with High-DPI Times New Roman legend.
    Uses super-sampling (scaling up) to make text sharp.
    Layout: 2 Rows (2 Cols)
    Legend Height: 40 (Base)
    Border: Thinner (1px)
    Position: Adjusted lower
    """
    # ---------------------------------------------------------
    # Part 1: 數據處理與 OpenCV 繪圖
    # ---------------------------------------------------------
    if torch.is_tensor(origin_img):
        img = origin_img.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)
    else:
        img = origin_img

    if torch.is_tensor(pred_prob):
        p = pred_prob.detach().cpu().squeeze().numpy()
    else:
        p = pred_prob
    p_bin = (p > threshold).astype(np.uint8) 

    if torch.is_tensor(true_mask):
        t = true_mask.detach().cpu().squeeze().numpy()
    else:
        t = true_mask
    t_bin = (t > 0.5).astype(np.uint8)

    if img.shape[0] != p_bin.shape[0] or img.shape[1] != p_bin.shape[1]:
        img = cv2.resize(img, (p_bin.shape[1], p_bin.shape[0]))

    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = p_bin.shape
    TP = (p_bin == 1) & (t_bin == 1)
    FP = (p_bin == 1) & (t_bin == 0)
    FN = (p_bin == 0) & (t_bin == 1)
    
    overlay = img_bgr.copy()
    overlay[TP] = [0, 255, 0]   
    overlay[FP] = [0, 0, 255]   
    overlay[FN] = [255, 0, 0]   
    
    alpha = 0.4
    combined = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    
    t_bin_cv = t_bin.astype(np.uint8)
    contours, _ = cv2.findContours(t_bin_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined, contours, -1, (0, 255, 255), 1) 

    # ---------------------------------------------------------
    # Part 2: Pillow 高解析度文字繪製
    # ---------------------------------------------------------
    
    # 1. 轉為 PIL 圖片
    pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    
    # ★★★ 放大圖片以提高文字 DPI ★★★
    scale_factor = 3
    
    w, h = pil_img.size
    new_w, new_h = w * scale_factor, h * scale_factor
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. 設置 Legend 區域參數
    legend_h = 40 * scale_factor       
    font_size = 13 * scale_factor      
    color_box_size = 12 * scale_factor 
    padding = 10 * scale_factor        
    
    # 3. 建立含 Legend 的新畫布
    new_img = Image.new('RGB', (new_w, new_h + legend_h), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    
    draw_legend = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            font_size
        )
    except IOError:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            font_size
        )
        print("Warning: Liberation Serif not found, using DejaVu Serif.")

    items = [
        ("TP (Hit)", (0, 255, 0)),
        ("FP (Over)", (255, 0, 0)),
        ("FN (Miss)", (0, 0, 255)),
        ("GT Boundary", (255, 255, 0))
    ]
    
    # 5. 排版 (2 cols x 2 rows)
    cols = 2
    step_x = new_w // cols
    row_height = legend_h // 2
    
    start_x = 19 * scale_factor
    
    # ★★★ [修改 1] 增加頂部間距 (5 -> 9.5)，讓圖示整體往下移 ★★★
    start_y_base = new_h + (9.5 * scale_factor) 
    
    for i, (text, color) in enumerate(items):
        row = i // cols
        col = i % cols
        
        x = start_x + col * step_x
        # 計算色塊的 Y 座標
        y = start_y_base + (row * row_height) - (5 * scale_factor) # 稍微往上拉一點以置中於 row
        
        # 畫色塊 (邊框 1 * scale)
        draw_legend.rectangle(
            [x, y, x + color_box_size, y + color_box_size], 
            fill=color, 
            outline="black", 
            width=1 * scale_factor
        )
        
        # 寫字
        text_x = x + color_box_size + (5 * scale_factor)
        
        # ★★★ [修改 2] 文字垂直位置微調 ★★★
        text_y = y - 3.8
        
        draw_legend.text((text_x, text_y), text, fill="black", font=font)

    new_img.save(save_path)

    intersection = np.sum(TP)
    sum_pred = np.sum(p_bin)
    sum_gt = np.sum(t_bin)
    calc_dice = (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)
    if sum_pred == 0 and sum_gt == 0: calc_dice = 1.0

    msg = (f"\n[Dice Diagnosis] {save_path.split('/')[-1]}\n"
           f"  Intersection(TP): {intersection}, Pred: {sum_pred}, GT: {sum_gt}\n"
           f"  Calculated Dice: {calc_dice:.4f}")
    
    return msg
