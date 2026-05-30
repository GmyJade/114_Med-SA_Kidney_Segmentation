# 114 Med-SA Kidney Segmentation
A medical image segmentation project for kidney region analysis using Med-SA.

## 📋 目錄

- [專案簡介](#-專案簡介)
- [環境需求](#-環境需求)
- [安裝方式](#-安裝方式)
- [資料集下載](#-資料集下載)
- [專案架構](#-專案架構)
- [使用方式](#-使用方式)
- [實驗結果](#-實驗結果)
- [參考資料](#-參考資料)

---

## 📌 專案簡介

<!-- 請在此說明這個專案的目標、使用的方法或模型（例如 U-Net、nnU-Net 等）、以及預期解決的問題 -->

---

## 🖥️ 環境需求

<!-- 請填入你的開發環境，例如： -->

- Python 3.10.6
- Ubuntu 20.04（或 22.04）
- CUDA 11.x（若使用 GPU）
- 相依套件詳見 `requirements.txt`

---

## ⚙️ 安裝方式

```bash
# 1. Clone 本專案
git clone https://github.com/你的帳號/你的專案名稱.git
cd 你的專案名稱

# 2. 安裝相依套件（若有）
pip install -r requirements.txt
```

---

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

## 🗂️ 專案架構

<!-- 可以用 tree 指令產生後貼上，例如：

```
專案名稱/
├── dataset/          # 資料集放置位置（不含於 repo）
├── src/              # 主要程式碼
├── configs/          # 設定檔
├── notebooks/        # Jupyter notebooks（若有）
├── requirements.txt
└── README.md
```
-->

---

## 🚀 使用方式

<!-- 說明如何執行訓練、測試、推論等，例如：

```bash
# 訓練
python train.py --config configs/default.yaml

# 測試
python test.py --checkpoint checkpoints/best.pth
```
-->

---

## 📊 實驗結果

<!-- 可以放表格、截圖或指標數值，例如：

| 模型 | Dice Score | HD95 |
|------|-----------|------|
| U-Net | 0.xx | xx |
| nnU-Net | 0.xx | xx |
-->

---

## 📚 參考資料

- [KiTS23 官方 GitHub](https://github.com/neheller/kits23)
- <!-- 其他參考論文或連結 -->
