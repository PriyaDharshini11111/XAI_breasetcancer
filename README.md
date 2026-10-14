# Explainable breast cancer diagnosis (TVGCF – OAST – DSORCNN – OOA)

Code for the manuscript *Enhancing Breast Cancer Diagnosis with Explainable AI*: TVGCF image enhancement, OAST
(orientation-guided attention Swin Transformer) lesion segmentation, DSORCNN (DenseNet-201 + multi-scale + residual) classification,
OOA hyper-parameter search, and Grad-CAM / LIME / SHAP explanations.

Archived release: **Zenodo DOI ``**

## Contents

| File | Purpose |
|---|---|
| `mainCode.ipynb` | Full pipeline (split → TVGCF → OAST → OOA → DSORCNN → test evaluation → efficiency → Grad-CAM/LIME/SHAP → optional 10-fold CV) |
| `main.py` | The same pipeline exported as a script (generated from the notebook) |
| `prepare_dataset.py` | Converts Mini-DDSM / CBIS-DDSM (via a CSV manifest) into the layout below, incl. outline→mask filling |
| `run_experiments.sh`, `aggregate_results.py` | Ablation variants × seeds, mean ± std, 95 % CI, ANOVA / Welch t-tests |
| `splits/<DATASET>_split.csv` | The exact train / validation / test assignment used in the paper (created on first run; commit it) |
| `requirements.txt`, `LICENSE` | Dependencies, MIT licence |

## Data

The datasets are public and are **not** redistributed here. Download them from the original sources and respect their terms of use and citation requirements.

| Name | Source | Notes |
|---|---|---|
| BUS (breast ultrasound images) | https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset (original data: Al-Dhabyani et al., *Data in Brief* 28:104863, 2020, https://doi.org/10.1016/j.dib.2019.104863) | licence: see the Kaggle page |
| Mini-DDSM | https://ardisdataset.github.io/MiniDDSM/ | Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)e |
| CBIS-DDSM | https://www.cancerimagingarchive.net/collection/cbis-ddsm/ (DOI 10.7937/K9/TCIA.2016.7O02S9CY) | CC BY 3.0 |

Expected layout (`DATA_ROOT` defaults to `./data`, or `/content/drive/MyDrive/breast_datasets` in Colab):

```
<DATA_ROOT>/BUS/benign|malignant|normal/<name>.png  (+ <name>_mask.png, extra masks <name>_mask_1.png …)
<DATA_ROOT>/MiniDDSM/benign|malignant|normal/...
<DATA_ROOT>/CBIS_DDSM/benign|benign_without_callback|malignant/...
```
BUS already has this layout. For Mini-DDSM and CBIS-DDSM build a manifest CSV (`image_path, class, mask_path`) from the dataset's metadata and run
`python prepare_dataset.py --manifest manifest.csv --out data/MiniDDSM --mask-mode outline` (`binary` for CBIS-DDSM ROI masks).

## Run

```bash
pip install -r requirements.txt
DATASET=BUS jupyter nbconvert --to notebook --execute mainCode.ipynb --output executed.ipynb     # or open the notebook in Colab / Jupyter
```
Settings are environment variables (or the configuration cell): `DATASET` (BUS | MiniDDSM | CBIS_DDSM), `VARIANT`
(full | no_tvgcf | no_oga | no_residual | no_ooa), `SEED`, `DATA_ROOT`, `OUT_ROOT`, `RUN_CV=1`, `RUN_XAI=0/1`, `PRETRAINED=0/1`.

Quick smoke test (minutes, CPU is enough): `SEG_EPOCHS=1 CLS_EPOCHS=1 OOA_POP=2 OOA_ITERS=1 OOA_PROXY_EPOCHS=1 RUN_XAI=0 PRETRAINED=0`

Several seeds and all ablation variants: `DATASET=BUS bash run_experiments.sh` (then `aggregate_results.py`).

## Outputs (`<OUT_ROOT>/<DATASET>/<VARIANT>/seed<SEED>/`)

`results.json` (all metrics and the configuration), `environment.txt` (exact package versions), `table_*.csv`, `fig_*.png`, `oast_best.pt`, `dsorcnn_best.pt`.

## Notes on the protocol

* Image-level stratified 70 / 10 / 20 split, created once and re-used from `splits/`. Patient IDs are not available, so the split is not patient-level; the near-duplicate diagnostic reports hash-identical images across splits.
* TVGCF, OAST, augmentation and all model selection use the training / validation images only; the test images are evaluated once.
* Swin-T and DenseNet-201 start from ImageNet weights (`timm`, `PRETRAINED=1`). Input size is 224 × 224 for every dataset.
* The OOA used here is a compact implementation (population 5, 3 iterations, 3 proxy epochs per evaluation; see `results.json → ooa_note`).
* 10-fold CV (`RUN_CV=1`) re-trains OAST inside each fold (`CV_RETRAIN_OAST=1`), so held-out folds never reach the segmentation network.
