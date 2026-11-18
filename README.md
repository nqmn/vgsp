# Variance-Guided Scalar Projection (VGSP)

**Python implementation for lightweight DDoS detection in Software-Defined Networks**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24183611.svg)](https://doi.org/10.5281/zenodo.24183611)

This repository contains `vgspv4.ipynb`, a Jupyter Notebook implementing **Variance-Guided Scalar Projection (VGSP)** for binary classification of normal and Distributed Denial-of-Service (DDoS) traffic in Software-Defined Networks (SDN). The notebook accompanies the research manuscript *A Variance-Guided Scalar Traffic Representation for Lightweight DDoS Detection in Software-Defined Networks*.

VGSP derives feature weights using a bounded **Class-Variance Ratio (CVR)** criterion and projects multiple traffic features onto a **single scalar representation** before classification. The research contribution is the CVR-guided scalar projection workflow, rather than treating the bounded CVR transformation in isolation as a new general-purpose separability metric.

## Research code and archive

- **Notebook:** [`vgspv4.ipynb`](vgspv4.ipynb)
- **Archived software DOI (Zenodo):** https://doi.org/10.5281/zenodo.24183611
- **Version represented here:** v4 (notebook last-update note: 13 November 2026)
- **Implementation:** Python / Jupyter Notebook

For scholarly citations, cite the version archived on **Zenodo** using the DOI above. The GitHub repository is intended for code access and future development; a subsequently updated GitHub file may differ from the archived Zenodo version.

## Method overview

For each feature \(j\), the implementation calculates within-class variance \(W_j\) and between-class variance \(B_j\), then computes

\[
\operatorname{CVR}(j)=\frac{B_j}{B_j+W_j+\varepsilon}, \qquad \varepsilon=10^{-10}.
\]

The highest-ranked features receive CVR-proportional projection weights. Their weighted combination produces a one-dimensional scalar \(z\), which is passed to a machine-learning classifier.

The notebook includes:

- Binary dataset loading and inspection, with duplicate-vector diagnostics.
- Stratified **80:20 development/test splitting** and **five-fold `StratifiedKFold`** evaluation.
- Training-fitted median imputation, zero-variance filtering where applicable, min-max scaling, CVR computation, feature ranking, and scalar projection.
- Feature-count ablation for **\(k=1,\ldots,12\)** using InSDN development-fold Macro-F1.
- Sensitivity analyses for `Protocol` encoding and correlated features.
- Comparisons with full-feature, variance-threshold, Fisher-score, PCA-1D, and LDA-1D baselines.
- Evaluation with Decision Tree (DT), Random Forest (RF), XGBoost (XGB), and LightGBM (LGBM).
- Predictive metrics, computational timings, paired Wilcoxon comparisons, visualization, and Decision Tree complexity analysis.

### Evaluation scope

In the archived v4 notebook, the InSDN development cross-validation selects **seven traffic features**: `Protocol`, `Down/Up Ratio`, `Init Bwd Win Byts`, `Pkt Len Min`, `ACK Flag Cnt`, `Bwd Pkt Len Min`, and `Bwd Pkts/s`. The selected **feature names** are then evaluated on DDOSDN2025 with dataset-specific preprocessing, CVR weights, and classifiers fitted from that dataset's own training partitions. **This is within-dataset evaluation, not a train-on-InSDN/test-on-DDOSDN2025 transfer experiment.**

The notebook uses stratified splitting, **not group-aware cross-validation**. Duplicate-vector hashes are calculated for diagnostic purposes; they are not used to create the folds. Fitted preprocessing and VGSP parameters are estimated from the appropriate training partition and applied unchanged to its validation or held-out test partition.

## Requirements

- Python 3.12 (notebook metadata records Python 3.12.3)
- JupyterLab or Jupyter Notebook
- NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, XGBoost, LightGBM, joblib, threadpoolctl, and IPython

Exact package versions are not pinned in the notebook. For precise environment-level reproduction, record the versions used in your own execution environment.

## Datasets and file paths

The notebook requires **two previously prepared binary CSV files** (labels: `0` = Normal, `1` = DDoS):

| Dataset | Expected CSV filename | Records in notebook | Numeric features used |
| --- | --- | ---: | ---: |
| InSDN | `insdn_ddos_binary_0n1d.csv` | 190,366 | 77 |
| DDOSDN2025 | `ddosdn2025_binary_0n1d.csv` | 200,378 | 77 |

**Before running the notebook, check or change `INSDN_PATH` and `DDOSDN_PATH` in the first configuration code cell.** The original settings are:

```python
INSDN_PATH = Path("../ds/insdn_ddos_binary_0n1d.csv")
DDOSDN_PATH = Path("../ds/ddosdn2025_binary_0n1d.csv")
```

If launching Jupyter from the repository directory, those settings correspond to the following example layout:

```text
workspace/
├── ds/
│   ├── insdn_ddos_binary_0n1d.csv
│   └── ddosdn2025_binary_0n1d.csv
└── VGSP/
    ├── README.md
    └── vgspv4.ipynb
```

Alternatively, place the CSV files in a local `data/` directory inside the repository and change the two paths to `Path("data/<filename>.csv")` accordingly. Relative paths are interpreted from **the Jupyter process working directory**, not necessarily from the notebook file's directory.

**Important:** The inputs are prepared *binary* CSVs, not necessarily the original raw dataset downloads. This notebook does not itself reconstruct those CSVs from the complete source datasets. Ensure that the files, labels, feature columns, and preprocessing match the research inputs before interpreting results. Data files are separately licensed by their original providers; the software license does not replace dataset licensing terms.

Dataset sources:

1. **InSDN:** Elsayed, M. S., Le-Khac, N.-A., and Jurcut, A. D. (2020), *InSDN: A Novel SDN Intrusion Dataset*, **IEEE Access**, 8, 165263–165284. https://doi.org/10.1109/ACCESS.2020.3022633
2. **DDOSDN2025:** Mokti, M. A. (2025), *DDOSDN2025*, Mendeley Data, Version 1. https://doi.org/10.17632/9c5yjbk354.1

## Limitations and reproducibility notes

- Experiments assess **binary Normal versus DDoS classification** on the specified, prepared datasets.
- The principal validation protocol is stratified and may not eliminate dependence among records sharing capture sessions, hosts, or identical feature vectors.
- The reported computational measurements are notebook-based experiments; they do not establish full packet-capture-to-controller operational latency.
- Results depend on input-data preparation, package versions, hardware, and execution environment. The presence of code in this repository does not by itself guarantee exact numerical reproduction on other systems.
- The uploaded v4 notebook uses **seven** selected features. Ensure that the associated manuscript, Zenodo version, and GitHub notebook describe the same released experiment.

## Citation

**Software** (confirm the exact author, title, and version against the published Zenodo citation metadata):

> Mokti, M. A. (2026). *Variance-Guided Scalar Projection (VGSP): Python Implementation for Lightweight DDoS Detection in Software-Defined Networks* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24183611

**Associated manuscript:**

> Mokti, M. A., Abu Bakar, K. A., and Zainol Ariffin, K. A. *A Variance-Guided Scalar Traffic Representation for Lightweight DDoS Detection in Software-Defined Networks*. Manuscript under revision; journal DOI not yet included in this repository.

Please cite the original InSDN and DDOSDN2025 sources when using those datasets.

## License

Use the **same software license** in GitHub and the Zenodo record. If you selected **MIT** in Zenodo, add a `LICENSE` file containing the MIT License text and the appropriate copyright holder(s) to this repository. Dataset licenses remain separate from the code license.

## Contact

**Mohd Adil Mokti**  
Center for Cyber Security, Faculty of Information Science and Technology, Universiti Kebangsaan Malaysia, Malaysia.
