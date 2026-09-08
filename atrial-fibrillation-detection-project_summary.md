# Atrial Fibrillation Detection

Team size: 2

## Overview

A 4-class arrhythmia classification project on short single-lead ECG recordings: **Normal (N)**, **Atrial Fibrillation (A)**, **Other rhythm (O)**, and **Noisy/unclassifiable (~)**. The repository does not contain a written problem statement or clinical-motivation narrative (no README), so the scope below is inferred directly from the code: given a ~9–60s single-lead ECG signal sampled at 300 Hz, the pipeline predicts one of the four rhythm classes, with Atrial Fibrillation detection as one of the four target outputs. The record naming convention (`A00001`…`A08528`), single-lead 300 Hz `.hea`/`.mat` WFDB-format files, and N/A/O/~ label scheme match the structure of a public short-single-lead-ECG classification dataset, but the dataset's official name/source/license is not stated anywhere in the repo — **[UNVERIFIED - check manually]**.

The project evolved through many iterations (raw-signal CNNs/TCNs, a Transformer, spectrogram CNNs, classical ML on hand-crafted features) before converging on a final two-stage hierarchical deep learning pipeline (see `Phase1_Combined_Data_Preparation.ipynb` → `Phase2_Stage1_Improved_BugFix_(2).ipynb` → `Phase3_Anora.ipynb` / `Phase3_Inference.ipynb`). Reference material is present in `BasePapers/` (3 PDFs) and `Papers/` (20 PDFs), indicating the design was informed by academic literature, though the specific papers used to justify specific design choices are not cited inline in the notebooks.

## Dataset

- Source files: `ECG_Dataset/` — single-lead ECG records in WFDB format (`.hea` header + `.mat` signal), e.g. `A00536.hea` header confirms **300 Hz** sampling rate, single lead.
- Working dataset: `ECG_Processed_Dataset/ecg_meta.csv` + `ecg_signals.npy`, loaded in `Phase1_Combined_Data_Preparation.ipynb`, printed as:
  - **8,528 total records** — label distribution: N=5,050, O=2,456, A=738, ~=284.
- Record-level train/test split (not window-level, to avoid leakage): `train_test_split(..., test_size=0.2, stratify=label, random_state=42)` → **6,822 train records / 1,706 test records**.
- Preprocessing (`Phase1_Combined_Data_Preparation.ipynb`): Butterworth band-pass filter 0.5–45 Hz (`scipy.signal.butter`, order 2, `filtfilt`) + per-window z-score normalization.
- Sliding-window extraction: window size 4,500 samples (15 s) at `FS=300`; 50% overlap for train windows (step 2,250), non-overlapping for test windows (step 4,500) → **22,083 train windows / 3,496 test windows** (printed per-class breakdown: train N=12,895 A=1,856 O=6,900 ~=432; test N=2,042 A=307 O=1,067 ~=80).
- Offline data augmentation (train only, minority classes A/O/~): CutMix-1D (random-segment swap between same-class windows, `alpha=0.3`) + additive Gaussian noise (σ = 0.01 × signal std), doubling each minority class → **31,271 train windows** after augmentation (N=12,895 A=3,712 O=13,800 ~=864).
- Feature engineering: 39 hand-crafted features per record via NeuroKit2 (`Old_Notebooks/09_NeuroKit.ipynb`) — RR-interval statistics (mean/std/min/max/CV/skew/kurtosis), HRV time-domain features (SDNN, RMSSD, pNN50, pNN20, etc.), and ECG morphology/R-amplitude statistics. Extraction succeeded for 8,519/8,528 records (9 failed); after dropping 2 rows with missing `rr_kurtosis`, 8,517 records remained for the classical-ML/MLP baselines. Features are broadcast per-window and scaled with `StandardScaler` fit on train only.
- Stage-2 (spectrogram) inputs: 3-channel 224×224 images built per non-Normal window — channel 0 = STFT log-magnitude, channel 1 = CWT (Morlet wavelet) magnitude, channel 2 = recurrence plot — generated only for non-Normal windows: **18,376 train / 1,454 test** spectrograms, stored as memory-mapped `.dat` files (train file logged as 11,064 MB, test as 875 MB).

## Models / architectures

**Final production pipeline — 2-stage hierarchical classifier (PyTorch):**

1. **Stage 1 — binary Normal vs Non-Normal** (`Phase2_Stage1_Improved_BugFix_(2).ipynb`): TCN-ResNet hybrid with NeuroKit2 feature fusion, **9,961,901 parameters** (printed via `sum(p.numel() ...)`).
   - Input conv: `Conv1d(1→64, kernel=15)` + PReLU
   - 5 dilated residual TCN blocks (kernel 9, weight-normalized, PReLU, dropout 0.3): dilations 1/2/4/8/16, channels 64→64→128→256→512→512
   - SE-1D channel-attention block (reduction 16) on the final 512-channel output
   - Multi-scale pooling: adaptive avg-pool + adaptive max-pool, concatenated (512+512=1024)
   - Signal head: `Linear(1024→256)` + LayerNorm + GELU + dropout
   - Feature branch: 39 NeuroKit2 features → `FeatureMLP` (`Linear(39→64)→LayerNorm→GELU→Dropout→Linear(64→64)→LayerNorm→GELU`)
   - Fusion classifier: `Linear(320→128)→LayerNorm→GELU→Dropout→Linear(128→2)`
   - Stochastic depth (LayerDrop, p=0.1) randomly skips TCN blocks during training
   - Loss: custom **Asymmetric Focal Loss** (`gamma_pos=1.0`, `gamma_neg=3.0`) to penalize missed Normals harder
   - Optimizer: AdamW (lr 3e-4, weight_decay 1e-4) with `OneCycleLR` (cosine anneal, 5% warmup)
   - `WeightedRandomSampler` for balanced batches; online augmentation (amplitude scale ±15%, ±2% circular time-shift, 5% time-mask), mixed-precision (`torch.amp`), gradient clipping (max norm 1.0), early stopping (patience 20, max 50 epochs)
   - Decision threshold tuned via a 0.25–0.75 sweep on the validation split; best result saved to `ECG_Final_Dataset/checkpoints_stage1_v2/best_threshold.json`: threshold=0.4, macro F1=0.9413, F1(Normal)=0.9332, F1(Non-Normal)=0.9494. (A separate saved threshold file used at Stage-3 inference time records threshold=0.3.)

2. **Stage 2 — 3-class subtype classifier (AF vs Other vs Noisy)**, applied only to windows Stage 1 flags Non-Normal: **EfficientNet-B2** (`torchvision.models.efficientnet_b2(weights=None)`, i.e. trained from scratch, not ImageNet-pretrained) on the 3-channel 224×224 spectrograms, with a custom classifier head. Two head variants appear in the repo: `Phase3_Anora.ipynb` uses `Dropout(0.4)→Linear(1408→3)`; `Phase3_Inference.ipynb` uses a larger head `Dropout(0.4)→Linear(1408→512)→BatchNorm1d→SiLU→Dropout(0.2)→Linear(512→3)`. Validation performance (`ECG_Final_Dataset/Anora_checkpoints_stage2/stage2_info.json`): macro F1 = 0.9681, F1(AF)=0.9656, F1(Other)=0.9880, F1(Noisy)=0.9508.

3. **Hierarchical combination** (`Phase3_Anora.ipynb`, `Phase3_Inference.ipynb`): every test window first passes through Stage 1; windows predicted Normal are finalized as Normal, windows predicted Non-Normal are routed to Stage 2 for the AF/Other/Noisy decision.

**Classical ML baselines on the 39 NeuroKit2 features** (`Old_Notebooks/09_NeuroKit.ipynb`), record-level 80/20 stratified split (6,813 train / 1,704 test), SMOTE oversampling on train only, `StandardScaler`:
- XGBoost (`n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`)
- Random Forest (`n_estimators=500, max_depth=20, min_samples_split=5, class_weight="balanced"`)
- Gradient Boosting (`n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8`)

**Residual MLP baseline on the same 39 features** (`Old_Notebooks/10_NeuroKit_DL.ipynb`): 3 residual blocks (256→256→128→64) with BatchNorm/ReLU/Dropout(0.3), embedding `Linear(39→256)`, classifier `Linear(64→4)` — **247,364 trainable parameters**. Trained with class-weighted, label-smoothed cross-entropy (weights N=1.0, A=3.0, O=2.5, ~=2.0; label_smoothing=0.1), AdamW + `ReduceLROnPlateau`, early stopping (patience 20, stopped at epoch 40 of 150 max).

**Earlier / superseded deep-learning iterations** (all in `Old_Notebooks/`, direct single-stage 4-class classification on raw signal or spectrograms, all Keras/TensorFlow unless noted):
- `06_AtrialFib_TCN.ipynb`: 3-block plain TCN (Conv1D, dilations 1/2/4, channels growing to 256), 9,000-sample (30 s) windows; final window-level validation accuracy 0.7465, test-set accuracy 0.75.
- `05_AtrialFib_TCNResNet.ipynb`: compares three variants on the same data — TCN-only, ResNet1D-only, and a dual-input TCN+ResNet hybrid that fuses raw-signal and spectrogram branches. Reported: TCN-only, ResNet-only, and Combined accuracies (combined model: accuracy 0.7095, weighted F1 0.6757).
- `07_AtrialFib_DataAug_TCNResnet.ipynb` / `08_AtrialFib_DataAug13026all_TCNResnet.ipynb`: Keras TCN (via `keras-tcn`, dilations up to 64) + ResNet1D hybrid with focal loss, trained on a fully class-balanced augmented set (13,028 windows/class, 52,112 train / 5,548 test windows). **7,986,066 total params** (`model.summary()` in notebook 08). Notebook 07 final: accuracy 0.8064, macro F1 0.7224; notebook 08 (larger balanced set) best val accuracy 0.8034. Checkpoints for this architecture are saved in the repo's `08_checkpoints/` (epoch-numbered `.keras` files).
- `11_Transformer.ipynb`: custom Vision-Transformer-style `ECGTransformer` operating directly on raw 4,500-sample windows split into 50 patches of 90 samples (300 ms) each, with a learned CLS token, sinusoidal positional encoding, `d_model=256`, 8 attention heads, 6 encoder layers, feed-forward dim 512 — **3,286,916 trainable params**. Final test accuracy 0.6802.
- `04_Atrilfib-Spectrogram.ipynb`: earlier exploration of spectrogram-based representations, precursor to the Stage-2 EfficientNet approach.
- `01_Initial.ipynb`, `02_Preprocessing.ipynb`, `Old_Phases/*.ipynb`: exploratory / draft versions of the data pipeline, superseded by `Phase1_Combined_Data_Preparation.ipynb`.

## Evaluation metrics

All numbers below are copied from notebook print output or from the checkpoint JSON files they write (`ECG_Final_Dataset/.../stage2_info.json`, `.../best_threshold.json`, `report_data.json`). Classical-ML/MLP metrics are on their own 1,704-sample record-level test split; the final hierarchical model's metrics are on the 3,496-window held-out test split described above (these are two different test sets and are not directly comparable).

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC | Notes |
|---|---|---|---|---|---|---|
| XGBoost (NeuroKit2 features) | 0.7729 | 0.6629 | 0.6871 | 0.6714 | not computed | `Old_Notebooks/09_NeuroKit.ipynb`; per-class F1: N 0.8552, A 0.7383, O 0.6478, ~ 0.4444 |
| Random Forest (NeuroKit2 features) | 0.7617 | 0.6435 | 0.6879 | 0.6573 | not computed | per-class F1: N 0.8505, A 0.7315, O 0.6341, ~ 0.4129 |
| Gradient Boosting (NeuroKit2 features) | 0.7635 | 0.6512 | 0.7058 | 0.6706 | not computed | per-class F1: N 0.8475, A 0.7303, O 0.6409, ~ 0.4636 |
| Residual MLP (NeuroKit2 features) | 0.6907 | 0.5876 | 0.6803 | 0.6168 | not computed | `Old_Notebooks/10_NeuroKit_DL.ipynb`; per-class F1: N 0.7747, A 0.6587, O 0.6034, ~ 0.4304 |
| Stage 1 (binary Normal/Non-Normal, TCN-ResNet+features), test set, thr=0.3 | 0.8518 | not computed | not computed | 0.8446 | not computed | `Phase3_Anora.ipynb`; confusion matrix TN=1867 FP=175 FN=343 TP=1111 |
| Stage 1 (binary), validation split, tuned thr=0.4 | not computed | not computed | not computed | 0.9413 | not computed | `best_threshold.json`; F1(Normal)=0.9332, F1(NonNormal)=0.9494 |
| Stage 2 (3-class AF/Other/Noisy, EfficientNet-B2), test set | 0.8776 | not computed | not computed | 0.7912 | not computed | `Phase3_Anora.ipynb` |
| Stage 2 (3-class), validation split | not computed | not computed | not computed | 0.9681 | not computed | `stage2_info.json`; F1(AF)=0.9656, F1(Other)=0.9880, F1(Noisy)=0.9508 |
| **Final hierarchical 4-class pipeline (test set, 3,496 windows)** | **0.8598** | **0.8147** | **0.7451** | **0.7720** | not computed† | `Phase3_Anora.ipynb`; MCC 0.7485; per-class F1: N 0.9225, A 0.7705, O 0.7548, ~ 0.6400 |

† `Phase3_Anora.ipynb` explicitly attempts a macro one-vs-rest AUC for the final 4-class result and it raises an error at runtime ("Target scores need to be probabilities for multiclass roc_auc..."), so no AUC value is produced by that run. `report_data.json` separately lists `"auc_ovr": 0.9413` under the overall summary, but that value is identical to the Stage-1 validation-split threshold-sweep macro F1 above and does not match any AUC computation actually executed in the notebooks — **[UNVERIFIED - check manually]**, treated here as not computed.

## Best-performing approach

The final two-stage hierarchical pipeline (TCN-ResNet + NeuroKit2 feature fusion for Normal/Non-Normal screening, followed by EfficientNet-B2 on 3-channel spectrograms for AF/Other/Noisy subtyping) is the approach carried into `Phase3_*` evaluation and is the only architecture with a full 4-class hierarchical evaluation. On the held-out 3,496-window test set it reached accuracy 0.8598 and macro F1 0.7720 (vs. 0.6573–0.6714 macro F1 for the classical-ML baselines and 0.6168 for the Residual MLP baseline, both on their own smaller test split). Within the notebook itself, this run explicitly checked its results against pre-set targets (`macro F1 ≥ 0.88` overall, per-class F1 targets of 0.88/0.80/0.70/0.80 for N/A/O/~) and reported **FAIL** on the AF (0.7705 vs 0.80 target), Noisy (0.6400 vs 0.80 target) classes and on the overall macro-F1 target, while passing on Normal (0.9225) and Other (0.7548).

## Techniques used

- Signal processing: Butterworth band-pass filtering (0.5–45 Hz), z-score normalization, STFT / continuous wavelet transform (Morlet) / recurrence-plot spectrogram generation.
- Feature engineering: NeuroKit2-derived RR-interval, HRV time-domain, and ECG-morphology features (39 total).
- Class-imbalance handling: SMOTE (classical ML baselines), `WeightedRandomSampler`, class-weighted cross-entropy, Asymmetric Focal Loss, and per-class multiplier-based offline oversampling with CutMix-1D + Gaussian noise (Stage 1 training data).
- Data augmentation: CutMix-1D and additive Gaussian noise (offline, minority classes only); amplitude scaling, circular time-shift, and time-masking (online, applied per-batch during Stage 1 training).
- Regularization: dropout, L2 weight decay/regularization, stochastic depth (LayerDrop), label smoothing (MLP baseline), gradient clipping.
- Model comparison / ablation: explicit TCN-only vs. ResNet-only vs. combined comparison (`05_AtrialFib_TCNResNet.ipynb`); explicit comparison across XGBoost/Random Forest/Gradient Boosting/Residual MLP baselines and the deep hierarchical pipeline.
- Hyperparameter/decision tuning: fine-grained probability-threshold sweep (0.25–0.75) to select the Stage 1 binary decision boundary that maximizes validation macro F1.
- Optimization/scheduling: AdamW, OneCycleLR (cosine, warmup), `ReduceLROnPlateau`, mixed-precision training (`torch.amp` and Keras `mixed_float16`), early stopping with patience, and checkpoint-resume logic for long (Google Colab) training sessions.
- Train/test methodology: record-level (not window-level) stratified splitting to avoid leakage between windows of the same recording.

No k-fold cross-validation was found executed in the repo — `StratifiedKFold`/`cross_val_score` are imported in `Old_Notebooks/09_NeuroKit.ipynb` but a corresponding call was not found in the extracted code/output, so this is **not** counted as a used technique.

## Tech stack

- Language: Python (Jupyter notebooks)
- Deep learning: PyTorch (+ `torchvision` for EfficientNet-B2), TensorFlow/Keras (+ `keras-tcn`) for earlier iterations
- Classical ML: scikit-learn (`RandomForestClassifier`, `GradientBoostingClassifier`, `StandardScaler`, `train_test_split`, metrics), XGBoost, imbalanced-learn (`SMOTE`)
- Signal/feature processing: NumPy, SciPy (`scipy.signal` — Butterworth filter), PyWavelets (`pywt` — CWT), scikit-image (`resize`), NeuroKit2 (ECG cleaning, R-peak detection, HRV features), `wfdb` (PhysioNet WFDB-format record reading)
- Data handling: pandas, `joblib` (model persistence), `pickle`
- Visualization: matplotlib, seaborn
- Utilities: tqdm (progress bars)
- Execution environments referenced in the notebooks: Google Colab (GPU training, Google Drive checkpoint storage) and Kaggle (`Phase3_Anora.ipynb` reads from `/kaggle/input/...`)
- No `requirements.txt`, environment YAML, or README was found in the repository; the local `ecg_env/` is a Python virtual environment (excluded from this analysis as it is installed third-party code, not project code).

## Quantifiable scope

- 8,528 total ECG records; 6,822 train / 1,706 test records (record-level 80/20 split).
- 22,083 train windows / 3,496 test windows before augmentation; 31,271 train windows after augmentation.
- 39 hand-engineered NeuroKit2 features per record.
- 18,376 train / 1,454 test 3-channel 224×224 spectrogram images generated for Stage 2 (train array logged at ~11,064 MB, test at ~875 MB on disk).
- At least 9 distinct trained models compared across the project's history: 4 classical/MLP baselines on tabular features (XGBoost, Random Forest, Gradient Boosting, Residual MLP) + at least 5 deep architectures on signal/spectrogram data (plain TCN, TCN-only/ResNet-only/combined comparison, augmented single-stage TCN-ResNet hybrid ×2 notebook variants, Transformer, and the final 2-stage TCN-ResNet+EfficientNet-B2 hierarchical pipeline).
- Model sizes (parameter counts, from printed `sum(p.numel()...)` or `model.summary()`): final Stage 1 TCN-ResNet+fusion model 9,961,901 params; earlier Keras TCN-ResNet hybrid 7,986,066 params; ECGTransformer 3,286,916 params; Residual MLP baseline 247,364 params.
- Logged training time: spectrogram generation for Stage 2 alone took ~1 hr 51 min for the 18,376-window train set and ~9 min 46 s for the 1,454-window test set (`tqdm` progress-bar output, `Phase1_Combined_Data_Preparation.ipynb`). Per-epoch/full end-to-end training time for the final PyTorch Stage 1/Stage 2 models is not logged in the retained notebook output — **not computed**.
