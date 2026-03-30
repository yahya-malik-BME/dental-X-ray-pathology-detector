# Dental X-Ray Pathology Detection — Project Writeup

## 1. Problem Statement

Dental pathologies such as cavities, root infections, and impacted teeth are among the most common conditions seen in clinical dentistry. Diagnosis relies heavily on manual interpretation of dental X-rays (radiographs) by trained clinicians — a process that is time-consuming, subjective, and prone to inter-observer variability. A single panoramic X-ray can contain 28–32 teeth, each requiring individual inspection for multiple possible conditions.

This project builds an **automated pathology detection system** that takes a dental X-ray image as input and outputs the precise location and type of every detected pathology. The goal is not to replace the clinician, but to serve as a second reader — flagging regions of concern that may be missed during routine examination, particularly in high-volume clinical settings or underserved regions with limited access to specialist radiologists.

---

## 2. What We Are Detecting

The system is trained to detect and localize four categories of dental pathology:

| Class | Clinical Description | Why It Matters |
|---|---|---|
| **Caries** | Tooth decay / cavity visible as a dark region on the enamel or dentin | Most common chronic disease worldwide; early detection prevents progression to pulp involvement |
| **Deep Caries** | Advanced decay that has extended close to or into the dental pulp | Requires urgent intervention (root canal or extraction); missed diagnosis leads to abscess |
| **Periapical Lesion** | Infection or abscess at the tip of the tooth root, visible as a dark radiolucent area | Indicates pulp necrosis; untreated cases can lead to systemic infection |
| **Impacted Tooth** | A tooth that has failed to erupt into its normal functional position, often blocked by bone or adjacent teeth | Common with third molars (wisdom teeth); can cause pain, cysts, and damage to neighboring teeth |

Each pathology is localized with a bounding box on the X-ray, so the clinician can see exactly where the model identified a problem.

---

## 3. Dataset — DENTEX 2023

We use the **DENTEX** (Dental Enumeration and Diagnosis) dataset, released as part of the MICCAI 2023 challenge. It is one of the largest publicly available annotated datasets for dental panoramic X-ray analysis.

### Dataset Statistics

| Metric | Value |
|---|---|
| Total images | 755 |
| Images with disease annotations | 678 |
| Background images (no pathology) | 77 |
| Total bounding box annotations | 3,529 |
| Average annotations per labeled image | 5.2 |
| Image type | Panoramic dental radiographs |
| Annotation source | Expert radiologists |
| Original format | COCO JSON |
| Converted format | YOLO (normalized center-x, center-y, width, height) |

### Class Distribution

| Pathology | Count | % of Total |
|---|---|---|
| Caries | 2,189 | 62.0% |
| Impacted tooth | 604 | 17.1% |
| Deep caries | 578 | 16.4% |
| Periapical lesion | 158 | 4.5% |

The dataset is **heavily imbalanced** — caries dominates at 62%, while periapical lesions make up less than 5%. This imbalance is clinically realistic (caries is far more prevalent than periapical pathology in the general population), but it presents a significant modeling challenge. We address this through class-weighted loss functions and augmentation strategies.

### Data Pipeline

The raw DENTEX annotations use a multi-level labeling scheme:
- `category_id_1`: Dental quadrant (1–4, dividing the mouth into upper-left, upper-right, lower-left, lower-right)
- `category_id_2`: Tooth number within the quadrant (1–8, using FDI notation)
- `category_id_3`: Disease label (the pathology we care about)

We extract only `category_id_3` (disease) and convert from COCO's absolute `[x_min, y_min, width, height]` format to YOLO's normalized `[center_x, center_y, width, height]` format. This conversion is necessary because YOLOv8 expects YOLO-format annotations during training.

---

## 4. Model Architecture

### Primary Model — YOLOv8m (Object Detection)

The core of the system is **YOLOv8m** (You Only Look Once, version 8, medium variant) from Ultralytics. YOLO is a single-stage object detection architecture — meaning it predicts bounding boxes and class labels in a single forward pass through the network, rather than using a two-stage approach (region proposal + classification).

#### Why YOLOv8?

1. **Speed**: Single-stage detection enables real-time inference, which is important for clinical deployment and the interactive Gradio demo
2. **Accuracy**: YOLOv8 achieves state-of-the-art results on COCO benchmarks, competitive with two-stage detectors like Faster R-CNN
3. **Anchor-free design**: Unlike earlier YOLO versions, YOLOv8 is anchor-free — it directly predicts the center and dimensions of bounding boxes rather than offsets from predefined anchor boxes. This simplifies training and improves generalization to objects of unusual aspect ratios (which dental pathologies often have)
4. **Built-in training pipeline**: Ultralytics provides a mature training loop with automatic learning rate scheduling, mosaic augmentation, and metric tracking

#### Why the "medium" variant?

YOLOv8 comes in five sizes: nano (n), small (s), medium (m), large (l), and xlarge (x). We chose **medium** as a balance between accuracy and computational cost. The medium variant has approximately 25.9 million parameters — large enough to learn the subtle radiographic features of dental pathologies, but small enough to train on a single GPU and run inference in near-real-time.

#### Architecture Overview

YOLOv8m consists of three major components:

1. **Backbone (CSPDarknet)**: A convolutional feature extractor that processes the input image (resized to 640x640) through a series of convolutional blocks with cross-stage partial connections. The backbone produces multi-scale feature maps at 1/8, 1/16, and 1/32 of the input resolution, capturing both fine-grained details (small caries) and large-scale context (jaw structure).

2. **Neck (PANet / FPN)**: A feature pyramid network that fuses the multi-scale feature maps from the backbone. This is critical for dental X-rays because pathologies vary dramatically in size — a small caries lesion may be 20x20 pixels while an impacted tooth region can span 200x200 pixels. The neck ensures the model can detect objects at all scales.

3. **Head (Decoupled Head)**: Separate prediction heads for classification and bounding box regression. The decoupled design allows each task to be optimized independently, which improves both localization accuracy and classification performance.

#### Transfer Learning

We initialize from **COCO-pretrained weights** (`yolov8m.pt`). Even though COCO contains natural images (cars, people, animals) and not dental X-rays, the low-level features learned on COCO (edges, textures, shapes) transfer surprisingly well to medical imaging. The model then fine-tunes all layers on our dental dataset, adapting the high-level features to recognize pathology-specific patterns.

### Secondary Model — EfficientNet-B3 (Classification)

As a complementary model, we include an **EfficientNet-B3** classifier. This is designed for a two-stage workflow:

1. YOLOv8 detects and crops individual tooth regions from the panoramic X-ray
2. EfficientNet-B3 classifies each cropped region into one of the 5 classes

EfficientNet uses compound scaling to balance network depth, width, and resolution, achieving high accuracy with fewer parameters than comparable architectures. The B3 variant has approximately 12 million parameters. We replace the original 1000-class ImageNet head with a 5-class dental pathology head and apply dropout (0.3) for regularization.

The classifier uses a **staged fine-tuning** strategy:
- **Epochs 1–5**: Backbone is frozen; only the new classification head is trained. This prevents catastrophic forgetting of useful ImageNet features while the randomly initialized head converges.
- **Epochs 6+**: Full backbone is unfrozen for end-to-end fine-tuning with a reduced learning rate (0.1x) to avoid destroying learned representations.

---

## 5. Training Strategy

### Optimization

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay; better generalization than vanilla Adam |
| Learning rate | 0.001 | Standard starting point for fine-tuning; decayed over training |
| Weight decay | 0.0005 | L2 regularization to prevent overfitting on a small dataset |
| LR schedule | Cosine annealing | Smooth decay from initial LR to 1e-6; avoids sharp drops |
| Warmup | 3 epochs | Gradually ramps up LR to avoid large early gradients that destabilize pretrained weights |
| Batch size | 16 | Fits in GPU memory for 640x640 images |
| Epochs | 100 (max) | Early stopping will likely trigger before 100 |
| Early stopping | patience=15 | If mAP does not improve for 15 consecutive epochs, training halts |

### Loss Function

YOLOv8 uses a **multi-task loss** with three components:

1. **Box loss (weight: 7.5)**: CIoU (Complete Intersection over Union) loss for bounding box regression. CIoU considers overlap area, center distance, and aspect ratio — all important for accurately localizing pathologies that vary widely in shape.

2. **Classification loss (weight: 0.5)**: Binary cross-entropy for class prediction. Lower weight than box loss because accurate localization is more clinically important than exact classification — a detected-but-misclassified lesion is still useful to the clinician.

3. **Distribution focal loss (weight: 1.5)**: Refines bounding box boundaries by predicting a probability distribution over possible offsets rather than a single regression value. This improves localization precision for ambiguous boundaries, which is common in X-rays where pathology edges blend into surrounding tissue.

### Data Splits

The dataset is split 70/15/15 into train, validation, and test sets:
- **Training (70%)**: Used for gradient updates
- **Validation (15%)**: Monitored during training for early stopping and checkpoint selection
- **Test (15%)**: Held out entirely; used only for final evaluation

The split is performed by shuffling filenames with a fixed random seed (42) for reproducibility.

---

## 6. Data Augmentation

Medical imaging augmentation requires special care — we must simulate realistic variations in imaging conditions without altering the clinical appearance of pathologies. An augmentation that makes a healthy tooth look diseased (or vice versa) would corrupt the training signal.

Our augmentation pipeline (applied to training images only):

| Augmentation | Parameters | Purpose |
|---|---|---|
| **CLAHE** | clip_limit=4.0, p=0.5 | Contrast-limited adaptive histogram equalization — critical for X-rays where important structures (bone vs. soft tissue) have low contrast |
| **Horizontal Flip** | p=0.5 | Dental anatomy is roughly symmetric; flipping doubles effective dataset size |
| **Brightness/Contrast** | limit=0.2, p=0.7 | Simulates variation in X-ray exposure settings across different machines and clinics |
| **Gamma Correction** | range=[80,120], p=0.5 | Simulates different detector response curves |
| **Gaussian Noise** | std=0.01-0.05, p=0.3 | Simulates sensor noise from different X-ray detector qualities |
| **Rotation** | limit=10 degrees, p=0.4 | Patients are not perfectly positioned; slight head tilt is common |
| **Blur** | kernel=3, p=0.2 | Simulates slight motion blur or focus variation |
| **Elastic Transform** | alpha=1, sigma=50, p=0.2 | Mild deformation to simulate natural anatomical variation |

**What we intentionally do NOT use:**
- Vertical flip (teeth don't appear upside down in clinical X-rays)
- Large rotations (>15 degrees would be unrealistic)
- Color jitter (X-rays are grayscale; artificial coloring would be meaningless)
- Cutout/erasing (could mask pathology regions and teach the model to ignore lesions)

Validation and test images receive only resize (to 640x640) and normalization — no augmentation, to ensure evaluation reflects real-world performance.

---

## 7. Preprocessing

Dental X-rays present unique preprocessing challenges:

1. **Grayscale to 3-channel**: X-rays are inherently grayscale, but our models (pretrained on RGB ImageNet/COCO data) expect 3-channel input. We replicate the grayscale channel 3 times. This preserves pretrained feature compatibility without losing information.

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Applied before inference to enhance local contrast. Standard histogram equalization can over-amplify noise, but CLAHE operates on small tiles (8x8 grid) with a clip limit (4.0) that prevents excessive amplification. This makes pathologies more visually distinct from surrounding healthy tissue — particularly important for subtle periapical lesions that appear as faint radiolucencies.

3. **Resize to 640x640**: YOLOv8 requires square input. Panoramic X-rays are typically wide (e.g., 2744x1316), so resizing introduces aspect ratio distortion. This is an accepted trade-off — the model learns to handle it during training. An alternative would be letterboxing (padding with black borders), which preserves aspect ratio but wastes input resolution.

4. **ImageNet normalization**: Pixel values are normalized using ImageNet mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]). Since our models are pretrained on ImageNet, matching the input distribution ensures the pretrained features remain meaningful.

---

## 8. Evaluation Metrics

We evaluate with metrics standard in both object detection research and medical imaging:

### Detection Metrics

- **mAP@0.50**: Mean Average Precision at IoU threshold 0.50. A detection is counted as correct if its bounding box overlaps with the ground truth by at least 50%. This is the primary metric — it captures both localization and classification accuracy.

- **mAP@0.50:0.95**: Mean AP averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05. This is a stricter metric that rewards precise localization. Lower scores here (compared to mAP@0.50) indicate the model detects pathologies in roughly the right area but struggles with exact boundaries.

- **Precision**: Of all predicted pathologies, what fraction are actually real pathologies? High precision means few false alarms.

- **Recall**: Of all real pathologies in the image, what fraction did the model detect? High recall means few missed diagnoses. **In clinical settings, recall is more important than precision** — a missed lesion is more dangerous than a false positive that a clinician can quickly dismiss.

- **Per-class AP**: Broken down by pathology type, because performance varies significantly across classes. We expect lower AP for periapical lesions (only 158 training examples, subtle appearance) and higher AP for impacted teeth (large, distinctive appearance).

### Classification Metrics (for the EfficientNet classifier)

- **Accuracy**: Overall correct classification rate
- **Per-class precision, recall, F1**: Reported via sklearn's classification report
- **Confusion matrix**: Saved as a PNG to identify systematic misclassification patterns (e.g., caries vs. deep caries confusion is expected and clinically meaningful)

---

## 9. System Architecture

```
                    +-----------------+
                    |  Dental X-ray   |
                    |   (input image) |
                    +--------+--------+
                             |
                    +--------v--------+
                    |  Preprocessing  |
                    |  CLAHE + Resize |
                    |  + Normalize    |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+        +----------v----------+
    |    YOLOv8m        |        |   EfficientNet-B3   |
    |   (detection)     |        |  (classification)   |
    |                   |        |                     |
    |  Outputs:         |        |  Input: cropped     |
    |  - bounding boxes |------->|    tooth regions    |
    |  - class labels   |        |  Output: pathology  |
    |  - confidence     |        |    class + conf     |
    +---------+---------+        +----------+----------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v--------+
                    |  Post-processing|
                    |  NMS + threshold|
                    |  + pathology    |
                    |    summary      |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+        +----------v----------+
    |   Annotated Image |        |   Clinical Report   |
    |   (bounding boxes |        |   (pathology count, |
    |    + labels)      |        |    per-class summary)|
    +-------------------+        +---------------------+
```

The primary workflow uses YOLOv8m alone for end-to-end detection. The EfficientNet classifier is available as a secondary verification step — crop each detected region and reclassify it for higher confidence in the diagnosis.

---

## 10. Interactive Demo

The project includes a **Gradio web interface** that makes the model accessible to non-technical users:

1. User uploads a panoramic or periapical dental X-ray
2. The image is preprocessed (CLAHE + resize) and passed through the YOLOv8m model
3. The output shows:
   - The original X-ray with color-coded bounding boxes drawn around detected pathologies
   - A summary table listing the count of each pathology type found
   - A confidence threshold (default 25%) that filters low-confidence predictions

Each pathology class has a distinct color for easy visual identification:
- Light blue: tooth (healthy)
- Amber: caries
- Red: deep caries
- Purple: periapical lesion
- Green: impacted tooth

The demo runs locally at `localhost:7860` and is designed for CPU inference, making it portable for presentations and demonstrations without requiring a GPU.

---

## 11. Experiment Tracking

All training runs are logged to **Weights & Biases (W&B)**, which tracks:

- Loss curves (box, classification, DFL) over epochs
- Validation metrics (mAP, precision, recall) over epochs
- Learning rate schedule
- Sample predictions on validation images every 10 epochs
- Full hyperparameter configuration for reproducibility
- System metrics (GPU utilization, memory usage)

This enables comparing different experiments — for example, YOLOv8s vs. YOLOv8m, or the effect of different augmentation strategies — in a structured, visual dashboard.

---

## 12. Addressing Dataset Challenges

### Class Imbalance

Periapical lesions make up only 4.5% of annotations. We mitigate this through:
- **Class weights** in the loss function (periapical lesion weight: 3.0x vs. caries: 2.5x)
- **Augmentation** that increases effective dataset size for all classes
- **Evaluation by per-class AP** to ensure minority classes are not ignored in aggregate metrics

### Small Dataset Size

755 images is small by deep learning standards. We address this through:
- **Transfer learning** from COCO-pretrained weights (the model starts with strong general features)
- **Conservative augmentation** that creates meaningful variations without introducing artifacts
- **Early stopping** to prevent overfitting (the model stops training when validation performance plateaus)
- **Weight decay** (L2 regularization) to constrain model complexity

### Ambiguous Boundaries

Pathology boundaries in X-rays are often diffuse — a caries lesion fades gradually into healthy dentin rather than having a sharp edge. We use:
- **Distribution focal loss** (DFL) which predicts a distribution over possible box boundaries rather than a single value
- **CIoU loss** which optimizes for overlap, center alignment, and aspect ratio simultaneously
- **mAP@0.50** as the primary metric (rather than mAP@0.50:0.95), acknowledging that clinical utility does not require pixel-perfect boundaries

---

## 13. Reproducibility

Every aspect of the pipeline is designed for full reproducibility:

- **Fixed random seeds** (seed=42) set across Python, NumPy, PyTorch, and CUDA
- **Deterministic CUDA operations** enabled via `torch.backends.cudnn.deterministic = True`
- **Data splits saved to disk** as text files listing filenames per split
- **All hyperparameters externalized** to YAML config files — no magic numbers in source code
- **W&B logging** captures the exact configuration used for every training run
- **Git-tracked codebase** with pinned dependency versions

To reproduce any experiment: check out the commit, install dependencies, and run `python -m src.train` with the same config files.

---

## 14. Tech Stack Summary

| Component | Technology | Why |
|---|---|---|
| Deep learning | PyTorch 2.x | Industry standard; strong ecosystem for research |
| Object detection | YOLOv8 (Ultralytics) | State-of-the-art speed/accuracy; mature training pipeline |
| Classification | EfficientNet-B3 (torchvision) | High accuracy with compound scaling; efficient parameters |
| Augmentation | Albumentations | Fast; supports bounding box transforms; medical imaging friendly |
| Experiment tracking | Weights & Biases | Visual dashboard; hyperparameter comparison; team collaboration |
| Config management | Hydra + OmegaConf | Type-safe YAML configs; command-line overrides; composable |
| Data validation | Pydantic | Runtime type checking for config objects |
| Demo interface | Gradio | Zero-frontend-code web UI; easy to share and present |
| Testing | pytest | Synthetic fixtures; fast CPU-only test suite |
| Linting | ruff + black + mypy | Consistent code style; type safety |

---

## 15. Clinical Disclaimer

This system is a **research tool** developed for academic purposes. It is not FDA-approved, CE-marked, or validated for clinical diagnostic use. Predictions should be interpreted by qualified dental professionals and should never be used as the sole basis for clinical decisions. The model's performance is bounded by the size and diversity of the DENTEX training dataset and may not generalize to X-rays from different imaging equipment, patient populations, or pathology types not represented in the training data.

---

## References

1. Hamamci, I.E., et al. "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays." *arXiv preprint arXiv:2305.19112* (2023).
2. Jocher, G., et al. "Ultralytics YOLOv8." https://github.com/ultralytics/ultralytics (2023).
3. Tan, M., & Le, Q.V. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML* (2019).
4. Buslaev, A., et al. "Albumentations: Fast and Flexible Image Augmentations." *Information* 11.2 (2020).
