<div align="center">

# 🔥 Laser-clad

### Process-aware melt-pool detection and temporal geometry prediction for laser cladding

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#installation)
[![YOLO](https://img.shields.io/badge/Detector-YOLO-orange.svg)](#method-overview)
[![LSTM](https://img.shields.io/badge/Predictor-LSTM-green.svg)](#method-overview)
[![TensorFlow](https://img.shields.io/badge/Backend-TensorFlow%2FKeras-ff6f00.svg)](#installation)
[![Interface](https://img.shields.io/badge/Interface-Tkinter-lightgrey.svg)](#graphical-interface)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple.svg)](#notes)

<p align="center">
  <em>A lightweight research codebase for extracting in-situ melt-pool visual descriptors and predicting laser-cladding geometric responses through a YOLO–LSTM pipeline.</em>
</p>

</div>

---

## 📌 Overview

**Laser-clad** provides an integrated inference and visualization pipeline for **laser cladding process analysis**. The repository combines a YOLO-based melt-pool detector with an LSTM-based temporal prediction model to estimate key forming-quality indicators from sequential melt-pool images and process metadata.

The pipeline is designed around a practical research scenario: given an ordered image sequence captured during laser cladding, the system first localizes the melt pool, extracts visual and process-aware descriptors, and then predicts the subsequent geometric responses through a temporal model.

<p align="center">
</p>

---

## ✨ Highlights

- 🔍 **Melt-pool localization** using a pretrained YOLO detector.
- 📐 **Process-aware feature extraction** from image content and filename-encoded process parameters.
- 🧠 **Temporal prediction** with an LSTM model using sliding-window sequence modeling.
- 🖥️ **Interactive visualization interface** for image navigation, detection display, and prediction inspection.
- 📦 **Ready-to-use model assets** including detector weights, LSTM model, scalers, and example validation image packages.

---

## 🧩 Method Overview

The overall workflow follows a detection-to-prediction paradigm:

```text
Input image sequence
        │
        ▼
YOLO-based melt-pool detection
        │
        ├── Melt-pool length  (MPL)
        ├── Melt-pool width   (MPW)
        ├── Average luminance (AVL)
        ├── Maximum luminance (MXL)
        ├── Laser power       (parsed from filename)
        └── Layer number      (parsed from filename)
        │
        ▼
Six-dimensional process-aware feature sequence
        │
        ▼
Feature normalization + sliding temporal window
        │
        ▼
LSTM-based temporal prediction
        │
        ├── Dilution ratio
        ├── Cladding depth
        └── Cladding height
```

This design enables the model to couple **melt-pool morphology**, **thermal-intensity cues**, and **process parameters** for temporal prediction of laser-cladding outcomes.

---

## 📁 Repository Structure

```text
Laser-clad/
├── YOLO-LSTM.py                  # Integrated YOLO detection, LSTM prediction, and GUI visualization
├── exp29/
│   └── weights/
│       └── best.pt               # Pretrained YOLO detector weights
├── trained_model/
│   ├── config.json               # LSTM configuration
│   ├── model.h5                  # Trained LSTM model
│   ├── feature_scaler.pkl        # Input feature scaler
│   └── target_scaler.pkl         # Output target scaler
├── Visualization Interface.png   # Graphical interface screenshot
├── val-image1.zip                # Validation image package
└── val-image2.zip                # Validation image package
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/fkj-art/Laser-clad.git
cd Laser-clad
```

### 2. Create a virtual environment

```bash
conda create -n laser-clad python=3.9 -y
conda activate laser-clad
```

or

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pillow joblib matplotlib scikit-learn tensorflow ultralytics
```

> **Note.** GPU acceleration is optional. For GPU inference, install a TensorFlow version compatible with your CUDA/cuDNN environment.

---

## 🚀 Quick Start
### 1. Download and merge test image folders
Due to file size limits, the test images are split into two packages. Please download both `val-image1.zip` and `val-image2.zip` from the repository, then extract and merge all images into a single folder:
https://github.com/fkj-art/Laser-clad
1. Download [`val-image1.zip`](./val-image1.zip) and [`val-image2.zip`](./val-image2.zip).
2. Extract the contents of both archives.
3. Place all extracted image files into one common folder (e.g., `val-image/`).

### 2. Check model paths

Before running the program, open `YOLO-LSTM.py` and make sure the YOLO weight path points to your local file.

The current detector path can be replaced by a relative path:

```python
YOLO_MODEL_PATH = r"./exp29/weights/best.pt"
LSTM_MODEL_DIR = "trained_model"
```

### 3. Run the graphical interface

```bash
python YOLO-LSTM.py
```

### 4. Select an image folder

After the GUI starts, click **Select Image Folder** and choose a folder containing sequential melt-pool images.

Supported image formats:

```text
.png, .jpg, .jpeg, .bmp, .tiff
```

The program will automatically:

1. sort images by filename;
2. detect the melt pool using YOLO;
3. extract six-dimensional frame-level features;
4. construct temporal windows for LSTM inference;
5. display detection boxes and prediction results in the GUI.

---

## 🗂️ Data Naming Convention

The current implementation parses **layer number** and **laser power** directly from the image filename.

Expected filename format:

```text
<layer>-<laser_power>-<sample_id>.<extension>
```

Example:

```text
1-1000-2519023 (1).png
```

This filename will be parsed as:

```text
Layer number = 1
Laser power  = 1000
```

If your filenames follow a different convention, modify the `parse_filename()` function in `YOLO-LSTM.py`.

---

## 📊 Input and Output Specification

### Input features

For each valid frame, the system constructs a six-dimensional feature vector:

| Feature | Description |
|---|---|
| `MPL` | Melt-pool length extracted from the detected bounding box |
| `MPW` | Melt-pool width extracted from the detected bounding box |
| `AVL` | Average luminance of the input image |
| `MXL` | Maximum luminance of the input image |
| `Laser Power` | Laser power parsed from filename |
| `Number of Layers` | Layer number parsed from filename |

### Predicted targets

The LSTM model predicts three geometric/process-quality indicators:

| Target | Description |
|---|---|
| `Dilution ratio` | Predicted dilution ratio |
| `Depth` | Predicted cladding depth |
| `Height` | Predicted cladding height |

---

## 🧠 Model Configuration

The released LSTM configuration is stored in `trained_model/config.json`:

```json
{
  "look_back": 2,
  "num_features": 6,
  "num_outputs": 3,
  "input_shape": [2, 6],
  "model_version": "1.0"
}
```

This means that each prediction is made from a temporal window containing two previous valid frames. Therefore, the first `look_back` valid frames do not have LSTM predictions.

---

## 🖥️ Graphical Interface

The GUI provides an end-to-end visualization environment for qualitative inspection:

- display the selected melt-pool image;
- overlay YOLO detection boxes;
- show parsed process parameters;
- report extracted visual descriptors;
- display predicted dilution ratio, depth, and height;
- navigate through image sequences using previous/next buttons.

This interface is intended for rapid experimental inspection and demonstration of process-aware temporal prediction results.

---

## 🔬 Reproducibility Notes

To improve reproducibility when reporting experimental results, please record the following settings:

- camera and image acquisition conditions;
- image resolution and preprocessing operations;
- filename convention for process parameters;
- YOLO detector weights and training data version;
- LSTM model version and scaler files;
- validation sequence split and ordering rule;
- hardware environment and software versions.

The current public release focuses on inference and visualization. For formal benchmarking, it is recommended to additionally provide training scripts, evaluation scripts, fixed random seeds, and detailed train/validation/test splits.

---

## 🧪 Suggested Evaluation Protocol

For academic reporting, the following protocol is recommended:

1. Use a fixed validation or test sequence split.
2. Report detection quality for melt-pool localization if bounding-box annotations are available.
3. Report regression metrics for each predicted target, such as MAE, RMSE, MAPE, and R².
4. Compare YOLO–LSTM with non-temporal or non-visual baselines.
5. Provide residual visualization for difficult samples, especially near layer transitions, local height fluctuations, and strong thermal-accumulation regions.

---

## 🛠️ Common Issues

### `YOLO model file not found`

Check whether `YOLO_MODEL_PATH` points to the correct `.pt` file:

```python
YOLO_MODEL_PATH = r"./exp29/weights/best.pt"
```

### `Filename format error`

Make sure the filename starts with layer number and laser power:

```text
1-1000-xxx.png
```

or modify `parse_filename()` according to your own dataset naming rule.

### `Insufficient Samples`

The LSTM model requires at least `look_back` valid frames. With the released configuration, at least two valid frames are needed before prediction can be generated.

### No melt pool detected

Check whether the input images are from the same domain as the detector training data. Large changes in illumination, camera angle, scale, or melt-pool appearance may reduce detection robustness.

---

## 📌 Notes

This repository is released as a **research prototype** for academic use. The current implementation is intended for offline analysis and visualization, rather than safety-critical closed-loop process control.

Recommended future improvements include:

- replacing hard-coded local paths with command-line arguments;
- adding `requirements.txt` or `environment.yml`;
- adding quantitative evaluation scripts;
- adding training scripts for detector and LSTM models;
- adding a formal license file;
- adding example outputs and benchmark tables.

---

## 📖 Citation

If this repository is helpful for your research, please consider citing the corresponding paper or this codebase.

```bibtex
@misc{laserclad_yololstm_2026,
  title  = {Laser-clad: Process-aware Melt-pool Detection and Temporal Geometry Prediction for Laser Cladding},
  author = {Kaijun,Fan; Yongjun,Shi},
  year   = {2026},
  url    = {https://github.com/fkj-art/Laser-clad}
}
```

Please update the BibTeX entry after the associated manuscript is accepted or officially published.

---

## 🙏 Acknowledgements

This project builds upon the deep-learning and scientific-computing ecosystem, including YOLO-based object detection, TensorFlow/Keras sequence modeling, and Python-based visualization tools.

---

## 📬 Contact

For questions, suggestions, or collaboration, please open an issue in this repository.

