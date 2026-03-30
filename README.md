# Dental X-Ray Pathology Detector

A deep learning system for automated detection and localization of dental pathologies in panoramic and periapical X-ray images. Built using YOLOv8 and PyTorch, trained on the DENTEX 2023 dataset.

> **Research collaboration project** — developed to support AI-assisted dental diagnostics research.

---

## What It Detects

| Pathology | Description |
|---|---|
| Caries | Tooth decay / cavity |
| Deep caries | Cavity extending near the pulp |
| Periapical lesion | Infection or abscess at the root tip |
| Impacted tooth | Tooth unable to erupt into normal position |

---

## Demo

```bash
python app/demo.py
# Open http://localhost:7860
```

Upload any panoramic dental X-ray and receive an annotated image with detected pathologies and a summary table.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yahya-malik-BME/dental-X-ray-pathology-detector.git
cd dental-X-ray-pathology-detector
pip install -r requirements.txt
```

### 2. Download the DENTEX dataset

```bash
python -c "from src.dataset import DENTEXDownloader; DENTEXDownloader('data/').download()"
```

### 3. Train

```bash
python -m src.train
```

Training logs to Weights & Biases. Set your W&B credentials first:
```bash
wandb login
```

### 4. Evaluate

```bash
python -m src.evaluate --weights outputs/weights/best.pt --split test
```

### 5. Run inference on a single image

```bash
python -m src.predict --image path/to/xray.jpg --weights outputs/weights/best.pt
```

---

## Project Structure

```
dental-X-ray-pathology-detector/
├── src/
│   ├── dataset.py      # Data loading and augmentation
│   ├── model.py        # YOLOv8 and EfficientNet model factory
│   ├── train.py        # Training loop with W&B logging
│   ├── evaluate.py     # mAP, precision/recall, confusion matrix
│   └── predict.py      # Single image and batch inference
├── configs/            # YAML config files (hyperparameters, paths, augmentation)
├── app/
│   └── demo.py         # Gradio interactive demo
├── notebooks/
│   ├── 01_eda.ipynb    # Exploratory data analysis
│   └── 02_eval.ipynb   # Post-training error analysis
└── tests/              # pytest test suite
```

---

## Results

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv8m | DENTEX test | — | — | — | — |

*Results will be updated after training on full dataset.*

---

## Dataset

**DENTEX (2023)**
- 4,000+ annotated panoramic dental X-rays
- Expert radiologist annotations in COCO format
- 4 pathology classes + tooth enumeration (FDI notation)
- Source: https://github.com/ibrahimethemhamamci/DENTEX

---

## Tech Stack

- **PyTorch 2.x** — core deep learning framework
- **YOLOv8** (Ultralytics) — real-time object detection
- **Albumentations** — medical imaging augmentation
- **Weights & Biases** — experiment tracking
- **Gradio** — demo interface
- **Hydra** — config management

---

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src
ruff check src/
```

---

## Citation

If you use this code in your research, please cite the DENTEX dataset:

```bibtex
@article{hamamci2023dentex,
  title={DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author={Hamamci, Ibrahim Ethem and ...},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
