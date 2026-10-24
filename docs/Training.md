# PlantScan AI — Model Training, Evaluation & Optimization Guide

**Version:** 1.0.0  
**Target Architecture:** MobileNetV2 Transfer Learning  
**Target Runtimes:** Edge TFLite (Mobile) & Cloud FastAPI (Inference Server)  
**Artifact Directory:** `ml/weights/<crop>/`  

---

## 1. Overview & Architectural Philosophy

PlantScan AI employs a **per-crop specialized classification topology** built on **MobileNetV2**. Instead of a monolithic 46-class network which suffers from severe cross-crop confusion (e.g., misclassifying potato early blight as tomato early blight), PlantScan AI routes inference to high-precision, crop-specific models guided by crop detection or user crop hints.

### Why MobileNetV2?
- **Inverted Residuals & Linear Bottlenecks**: Drastically reduces parameter footprint while maintaining high gradient representation.
- **Edge Efficiency**: Target mobile footprint is under **15 MB** per crop model with sub-50ms CPU inference on ARM mobile processors.
- **Pretrained Representations**: Leverages ImageNet features for rapid convergence even on small specialized agricultural datasets.

---

## 2. Neural Network Architecture

### 2.1 Layer Topology
```
Input (224 x 224 x 3, RGB)
  │
  ▼
MobileNetV2 Feature Extractor (ImageNet weights, top stripped)
  │ (Frozen during Phase 1; top blocks fine-tuned in Phase 2)
  ▼
GlobalAveragePooling2D
  │
  ▼
Dropout (rate = 0.25 - 0.3)
  │
  ▼
Dense (num_classes, activation="softmax")
```

### 2.2 Unified Classification Head Standardization
> [!IMPORTANT]
> **Universal Softmax Formulation**: All crop heads (including binary 2-class datasets such as Grape) are strictly parameterized with `Dense(num_classes, activation="softmax")` and trained with `loss="sparse_categorical_crossentropy"`.
> 
> *Historical Post-Mortem*: Previous pipelines used `Dense(1, activation="sigmoid")` for binary datasets. In TFLite edge runtimes, `np.argmax(probs)` on a 1D scalar `[p]` always returned index `0`, making it impossible to predict the healthy class. Standardizing on `num_classes >= 2` outputs `[p_0, p_1]` where $p_0 + p_1 = 1$, enabling seamless `np.argmax(probs)` resolution across both cloud and mobile edge runtimes.

---

## 3. Training Hyperparameters & Optimization

| Hyperparameter | Value | Description |
|---|---|---|
| **Input Shape** | `(224, 224, 3)` | Standard MobileNetV2 spatial resolution |
| **Batch Size** | `16` or `32` | Adjusted based on sample volume per crop |
| **Optimizer** | `Adam(learning_rate=1e-4)` | Adaptive moment estimation |
| **Loss Function** | `SparseCategoricalCrossentropy` | `from_logits=False` (normalized probabilities) |
| **Epochs** | `15 - 20` | Controlled by EarlyStopping |
| **Early Stopping** | `patience=5, restore_best_weights=True` | Monitored on `val_loss` |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=2)` | Step decay when validation loss stagnates |
| **Dropout Rate** | `0.25` | Regularization before final projection layer |

---

## 4. Two-Phase Transfer Learning Strategy

### Phase 1: Feature Extraction (Freeze Backbone)
- The entire MobileNetV2 feature extractor is set to `trainable = False`.
- Only the top `GlobalAveragePooling2D`, `Dropout`, and `Dense(softmax)` layers are trained.
- Learning rate: $\eta = 10^{-4}$.
- Epochs: 10–12 epochs until classification head weights stabilize.

### Phase 2: Fine-Tuning (Selective Unfreeze)
- The top 30–40 convolutional layers of MobileNetV2 are unfrozen (`trainable = True`).
- Batch normalization layers remain frozen (`training = False`) to prevent internal covariate shift breakdown.
- Learning rate is reduced: $\eta = 10^{-5}$ with Adam.
- Epochs: 5–8 epochs to adapt high-level visual features to specific leaf lesion textures.

---

## 5. TFLite Conversion & Optimization

After Keras model checkpointing, models are converted to TensorFlow Lite format for deployment:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Optional float16 quantization for 50% storage reduction:
# converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("ml/weights/<crop>/model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Artifact Manifest Generation (`classes.json`)
Every training run automatically extracts `train_ds.class_names` and writes a corresponding `classes.json` adjacent to the `.tflite` model:

```json
[
  "grape_black_rot",
  "grape_healthy"
]
```

This guarantees 1:1 synchronization between model output tensor indices and canonical disease classes without relying on runtime folder discovery.

---

## 6. Evaluation Metrics & Verified Benchmarks

Evaluation is conducted on holdout test partitions using precision, recall, and Macro F1-score:

| Crop Model | Test Accuracy | Macro F1-Score | Model Output Shape | Weight Size (.tflite) |
|---|---|---|---|---|
| **Tomato** | **94.2%** | **0.938** | `[1, 10]` | ~8.9 MB |
| **Grape** | **78.6%** | **0.785** | `[1, 2]` | ~8.7 MB |
| **Apple** | **64.3%** | **0.638** | `[1, 3]` | ~8.7 MB |
| **Potato** | **88.5%** | **0.879** | `[1, 3]` | ~8.7 MB |
| **Rice** | **85.0%** | **0.842** | `[1, 6]` | ~8.8 MB |
| **Banana** | **89.1%** | **0.887** | `[1, 4]` | ~8.8 MB |
| **Mango** | **86.4%** | **0.859** | `[1, 8]` | ~8.9 MB |

---

## 7. Execution CLI & Reproducibility

### Train a Specific Crop Model
```bash
# Example: Retrain Grape model with standardized softmax head
python experiments/train_crop_mobilenetv2.py --crop grape --epochs 15 --batch-size 16

# Example: Retrain Apple model
python experiments/train_crop_mobilenetv2.py --crop apple --epochs 15 --batch-size 16
```

### Train All Crops in Batch Mode
```bash
python experiments/train_crop_mobilenetv2.py --crop all --epochs 15
```

### Run Model Evaluation Report Generator
```bash
python experiments/evaluate_crop_models.py --crop all
```
