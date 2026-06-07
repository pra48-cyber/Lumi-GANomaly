# Lumi-GANomaly

## 🔗 Dataset Link
The video frame sequences used for evaluating and training this framework can be accessed here:
* **Google Drive Dataset**: [Download Dataset Here](https://drive.google.com/drive/folders/1qCgN4ZYWda0hu5Et6RtNLmsj6GhEnjjk?usp=sharing)
---

## 📁 Repository Structure

```text
├── models/
│   └── anomaly_detector.py   # CBAM, MemoryModule, and Generator definitions
├── utils/
│   └── helpers.py            # Dataset wrapper, PIL conversions, and heatmap utilities
├── train.py                  # Production training script with Cosine Annealing LR
├── test.py                   # Evaluation script for raw anomaly score plotting
└── README.md                 # Project documentation
