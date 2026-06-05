# 114 Med-SA Kidney Segmentation
A medical image segmentation project for kidney region analysis using Med-SA.

## 📋 目錄

- [專案簡介](#專案簡介)
- [安裝方式](#安裝方式)
- [資料集下載](#資料集下載)
- [資料集前處理](#資料集前處理)
- [專案架構](#專案架構)
- [使用方式](#使用方式)
- [實驗結果](#實驗結果)
- [參考資料](#參考資料)

---
<a name="專案簡介"></a>
## 📌 專案簡介

本研究以醫學影像分割為主題，針對腹部 CT 影像中的腎臟相關區域進行自動化分割與量化分析。為改善 SAM 直接應用於醫學影像時適應能力不足的問題，本研究採用 Med-SA-Adapter 作為核心模型，透過 Adapter-based fine-tuning 降低模型微調成本，並提升其對醫學影像特徵的學習能力。

為符合現有臨床常用之腎臟體積估算流程，並降低完整 3D volume training 對 GPU 記憶體與運算資源的需求，本研究改採 2D slice-based multi-planar strategy，將 3D CT volume 分別轉換為 **axial、coronal 與 sagittal** 三個方向的 2D slices 進行訓練與比較。實驗使用 KiTS23 與 BTCV 資料集，並以 Dice coefficient 與 IoU 評估模型分割表現。

此外，本研究提出 **Maximum Slice Area** 計算流程，根據 prediction mask 自動計算各切片之前景面積，找出目標區域面積最大的代表性切片，作為後續腎臟面積分析與體積估算的前置依據。實驗結果顯示，本研究流程能完成腎臟相關區域分割，並自動提供最大面積切片資訊，可降低人工逐張檢視 CT 影像的負擔。

整體而言，本研究建立了一套結合三切面分割與最大面積切片判定的醫學影像分析流程，具備輔助臨床量化分析與後續系統擴充之應用潛力。

> **注意**：本專案以 **coronal 切面**與**KiTS23資料集**為實作範例。如需使用其他切面，可參考相同架構自行調整資料載入與訓練設定。


---

<a name="安裝方式"></a>
## ⚙️ 安裝方式

### 1. Clone 本專案

```bash
git clone https://github.com/GmyJade/114_Med-SA_Kidney_Segmentation.git
cd 114_Med-SA_Kidney_Segmentation/Medical_SAM_Adapter_Coronal
```

### 2. 建立 Conda 環境

```bash
conda env create -f environment.yml
conda activate sam_adapt
```

### 3. 下載 SAM 模型權重

本專案使用 [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) 的預訓練權重。

```bash
# 下載權重檔案
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 建立資料夾並移動檔案
mkdir -p ./checkpoint/sam
mv sam_vit_b_01ec64.pth ./checkpoint/sam/
```

完成後，確認檔案位於以下路徑：

```
Medical_SAM_Adapter_Coronal/
└── checkpoint/
    └── sam/
        └── sam_vit_b_01ec64.pth
```
---

<a name="資料集下載"></a>
## 📦 資料集下載

本專案使用 [KiTS23](https://github.com/neheller/kits23) 資料集進行腎臟影像分割實驗。
由於資料集檔案較大，不直接包含於此 repository，請依以下步驟自行下載。

### 安裝 KiTS23 官方套件

```bash
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
```

### 下載資料

安裝完成後，執行以下指令，資料集將自動下載至 `dataset/` 資料夾：

```bash
kits23_download_data
```

> **注意事項**
> - 官方建議使用 Python 3.10.6 與 Ubuntu 環境
> - 本專案不包含原始資料集，請依 [KiTS23 官方說明](https://github.com/neheller/kits23) 自行下載

---

<a name="資料集前處理"></a>
## 🔧 資料集前處理

下載完成後，需將 KiTS23 的原始格式轉換為 Medical-SAM-Adapter 可讀取的結構，並自動生成 `dataset.json`。

### 執行轉換腳本

```bash
cd 114_Med-SA_Kidney_Segmentation
python prepare_kits23_dataset.py
```

腳本會自動完成以下工作：

| 步驟 | 說明 |
|------|------|
| 建立資料夾 | 在目標路徑建立 `imagesTr/` 與 `labelsTr/` |
| 複製影像 | 將 `imaging.nii.gz` 複製並重新命名至 `imagesTr/` |
| 複製標籤 | 將 `segmentation.nii.gz` 複製並重新命名至 `labelsTr/` |
| 生成 JSON | 自動切分訓練／驗證集，輸出 `dataset.json` |

### 預設路徑設定

請確認 `prepare_dataset.py` 頂部的路徑設定與你的環境一致：

```python
source_dir = "./kits23/dataset"          # KiTS23 原始資料位置
target_dir = "./dataset/KiTS23_for_MSA"  # 輸出目標位置
```

### 輸出結構

轉換完成後，目標資料夾結構如下：

```
dataset/KiTS23_for_MSA/
├── imagesTr/
│   ├── case_00000.nii.gz
│   ├── case_00001.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_00000.nii.gz
│   ├── case_00001.nii.gz
│   └── ...
└── dataset.json
```

### 標籤定義

| Label ID | 類別 |
|----------|------|
| 0 | Background（背景） |
| 1 | Kidney（腎臟） |
| 2 | Tumor（腫瘤） |
| 3 | Cyst（囊腫） |

> **注意事項**
> - 預設將最後 **10 筆**作為驗證集，其餘為訓練集，可在腳本內調整 `val_count`
> - 若目標檔案已存在則自動跳過，重複執行不會覆蓋

---
<a name="專案架構"></a>
## 🗂️ 專案架構

```
114_Med-SA_Kidney_Segmentation/
├── prepare_kits23_data.py              # 將 KiTS23 原始資料轉換為 MSA 可讀格式
├── kits23_nifti_viewer.py              # NIfTI 影像視覺化工具
├── dataset/                            # 本地資料集目錄（未上傳至 GitHub）
├── README.md
├── LICENSE
└── Medical_SAM_Adapter_Coronal/        # Med-SA-Adapter 核心實作
    ├── train.py                        # 訓練腳本
    ├── val.py                          # 驗證腳本
    ├── function.py                     # 訓練與驗證函式
    ├── cfg.py                          # 參數設定
    ├── train_kits23_coronal.sh         # 訓練執行腳本（Shell）
    ├── val_kits23_coronal.sh           # 驗證執行腳本（Shell）
    ├── environment.yml                 # Conda 環境設定
    ├── dataset/
    │   └── kits.py                     # KiTS23 資料集載入器
    ├── models/                         # 模型架構
    ├── conf/                           # 全域設定檔
    ├── checkpoint/                     # SAM 預訓練權重（未上傳）
    ├── logs/                           # 訓練與驗證紀錄（未上傳）
    ├── figs/                           # 架構圖與實驗圖表
    ├── guidance/                       # Prompt guidance 相關
    └── pytorch_ssim/                   # SSIM 損失函式模組
```

---


<a name="使用方式"></a>
## 🚀 使用方式

### 訓練

確認資料集已依照前處理步驟準備完畢後，進入核心目錄並執行訓練腳本：

```bash
cd Medical_SAM_Adapter_Coronal
bash train_kits23_coronal.sh
```

或直接呼叫 Python 訓練腳本（可自行調整參數）：

```bash
python3 train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name kits23_Med-SA_train_coronal_EPOCH_500 \
    -encoder vit_b \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -image_size 1024 \
    -b 2 \
    -dataset kits \
    -data_path "放實際的資料集路徑" \
    -num_sample 4 \
    -vis 5 \
    -slice_plane coronal
```

### 驗證

```bash
bash val_kits23_coronal.sh
```

或手動執行：

```bash
python3 val.py \
    -net sam \
    -mod sam_adpt \
    -exp_name val_kits23_coronal_full_slices \
    -encoder vit_b \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -weights "放自己訓練好的 best dice 權重" \
    -image_size 1024 \
    -b 2 \
    -dataset kits \
    -data_path "放實際的資料集路徑" \
    -num_sample 4 \
    -vis 1 \
    -slice_plane coronal
```

### NIfTI 影像視覺化

可使用內建的視覺化工具檢視原始影像：

```bash
python kits23_nifti_viewer.py
```

> **注意**：訓練參數（batch size、epoch 數、learning rate 等）請依實際執行環境與 `train_kits23_coronal.sh` 內容調整。
---
<a name="實驗結果"></a>
## 📊 實驗結果
> 以下結果基於 KiTS23 資料集，訓練 500 epochs。
### 訓練曲線（Training Loss & Validation Metrics）

以下為三個切面於 KiTS23 資料集上訓練 500 epochs 的 Loss 與驗證指標曲線：

| 切面 | Best Dice | Best Dice Epoch | Best IoU | Best IoU Epoch |
|------|-----------|-----------------|----------|----------------|
| Axial | 0.9481 | 466 | 0.9016 | 465 |
| Coronal | 0.9233 | 256 | 0.8708 | 415 |
| Sagittal | 0.9174 | 391 | 0.8680 | 410 |

<img width="1260" height="396" alt="image" src="https://github.com/user-attachments/assets/c12069e2-26a5-4a98-beec-826273ac6db7" />
三個切面皆達到穩定收斂，其中 axial 切面取得最高的 Dice 與 IoU 分數。


---

### 分割結果視覺化（TP / FN / FP）

下圖以顏色標示各區域的分割正確性，三個切面皆能有效分割腎臟區域，主要錯誤集中於邊緣區域的 FN（漏分）與少量 FP（誤分）：

| 顏色 | 意義 |
|------|------|
| 🟩 綠色（TP） | 正確分割的腎臟區域 |
| 🟦 藍色（FN） | 漏分的腎臟區域 |
| 🟥 紅色（FP） | 誤判為腎臟的背景區域 |
| 🟨 黃色（GT） | Ground Truth 腎臟位置 |

<img width="1280" height="582" alt="image" src="https://github.com/user-attachments/assets/699384ba-d137-46fb-9086-b7a337b1b98a" />


---

### Maximum Slice Area 最大面積切片結果

根據 prediction mask 自動找出各切面面積最大的代表性切片，並與 Ground Truth 進行比對：

| 切面 | Epoch | Dice | IoU | 預測面積 (mm²) | GT 面積 (mm²) |
|------|-------|------|-----|----------------|---------------|
| Axial | 466 | 0.8991 | 0.9258 | 8991.7 | 8469.2 |
| Coronal | 256 | 0.9466 | 0.9633 | 11600.9 | 11575.8 |
| Sagittal | 391 | 0.9361 | 0.9585 | 5058.4 | 4942.8 |
<img width="3932" height="2030" alt="圖片5" src="https://github.com/user-attachments/assets/6927669a-6b43-4709-8445-59c729440960" />

---
<a name="參考資料"></a>
## 📚 參考資料
### 資料集
- [KiTS23 官方 GitHub](https://github.com/neheller/kits23)
- [Synapse BTCV Dataset](https://www.synapse.org/Synapse:syn3193805/files/)
- [KiTS23 Challenge 官網](https://kits-challenge.org/kits23/)

### 論文
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) — Kirillov et al., ICCV 2023
- [Medical SAM Adapter](https://doi.org/10.1016/j.media.2025.103547) — Wu et al., Medical Image Analysis 2025
