# Vision Transformer and CNN implementation for CIFAR 10 Classification and explainability

This project implements and compares two deep learning architectures — **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** — on the CIFAR-10 image classification benchmark. Both models are implemented **from scratch** using PyTorch. The project includes full training pipelines, model explainability (via Grad-CAM and self-attention visualizations), and experiment tracking with Weights & Biases.

---

## 🧠 Project Objectives

* Implement **CNN**, **ViT**, and **Attention Maps** entirely from scratch (no pretrained models)
* Train both architectures on CIFAR-10 using **PyTorch Lightning**
* Explain CNN decisions with **Grad-CAM**
* Visualize ViT attention mechanisms
* Evaluate performance using confusion matrices, classification reports, and accuracy
* Track experiments with **Weights & Biases (wandb)**

---

## 🖼 Dataset

* **CIFAR-10**: 60,000 32×32 color images in 10 classes (airplane, car, bird, etc.)
* Split: 50,000 for training, 10,000 for testing
* Images are normalized using CIFAR-10 statistics

---

## 📦 Dependencies

* `torch`, `torchvision`
* `pytorch-lightning`
* `wandb`
* `matplotlib`, `seaborn`, `scikit-learn`

Install everything via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Notebook

To train and evaluate the models:

```bash
jupyter notebook cnn_vit_cifar10.ipynb
```

Make sure to initialize Weights & Biases with your account:

```python
import wandb
wandb.login()
```

---

## 📊 Example Outputs

### ✅ Grad-CAM for CNNs

Visualizes the spatial regions the CNN relied on for prediction.

![image](https://github.com/user-attachments/assets/74ea9790-8045-43e8-8a1b-8f75444233b3)

### 🧭 Attention Maps in ViT

Illustrates self-attention distribution over image patches.

![image](https://github.com/user-attachments/assets/1992c7aa-b42a-4e60-84c3-c734ad951cda)


### 📉 Confusion Matrix

Evaluates class-level performance and misclassifications.

![image](https://github.com/user-attachments/assets/8c2e0c8f-1843-49e1-9385-4c6c792bd7d2)


---

## 📈 Results Summary

| Metric            | CNN                   | Vision Transformer      |
| ----------------- | --------------------- | ----------------------- |
| Accuracy (Test)   | \~83%                 | \~71%                   |
| Explainability    | Grad-CAM (local)      | Attention Maps (global) |
| Model Size        | Small (\~559K params) | Larger (\~2.7M params)  |
| Convergence Speed | Faster                | Slower                  |

---

## 🧑‍💻 Author

**Yarden Cohen**
M.Sc. Student in Data Science
Ben-Gurion University of the Negev
[LinkedIn](https://www.linkedin.com/in/yarden-cohen2/)


