import os
import nibabel as nib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

DATA_DIR = r"/home/user412771064/project/Medical_project/kits23/dataset"

FONT_NAME = "Liberation Serif"

plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["figure.titlesize"] = 20


@st.cache_data
def get_cases(data_dir):
    cases = sorted(
        case for case in os.listdir(data_dir)
        if os.path.exists(os.path.join(data_dir, case, "imaging.nii.gz"))
    )
    return cases


@st.cache_data
def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data


st.set_page_config(
    page_title="KiTS23 NIfTI Viewer",
    layout="wide"
)

st.title("KiTS23 NIfTI Viewer")

cases = get_cases(DATA_DIR)

if not cases:
    st.error(f"No imaging.nii.gz found under {DATA_DIR}")
    st.stop()

case_name = st.selectbox("Select case", cases)

nii_path = os.path.join(DATA_DIR, case_name, "imaging.nii.gz")
data = load_nii(nii_path)

z, y, x = data.shape

st.write(f"Current case: `{case_name}`")
st.write(f"Data shape: Z={z}, Y={y}, X={x}")

col_slider1, col_slider2, col_slider3 = st.columns(3)

with col_slider1:
    z_idx = st.slider("Axial Z", 0, z - 1, z // 2)

with col_slider2:
    y_idx = st.slider("Coronal Y", 0, y - 1, y // 2)

with col_slider3:
    x_idx = st.slider("Sagittal X", 0, x - 1, x // 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(data[z_idx], cmap="gray")
axes[0].set_title(f"Axial\n{case_name} | Z={z_idx}", fontsize=16, fontname=FONT_NAME)

axes[1].imshow(data[:, y_idx, :], cmap="gray")
axes[1].set_title(f"Coronal\n{case_name} | Y={y_idx}", fontsize=16, fontname=FONT_NAME)

axes[2].imshow(data[:, :, x_idx], cmap="gray")
axes[2].set_title(f"Sagittal\n{case_name} | X={x_idx}", fontsize=16, fontname=FONT_NAME)

for ax in axes:
    ax.tick_params(labelsize=10)

fig.suptitle(
    f"Case: {case_name}",
    fontsize=20,
    fontname=FONT_NAME,
    y=1.05
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.tight_layout()

st.pyplot(fig)