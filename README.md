# CIFAR-10: NdLinear vs Baseline CNN

This project compares a traditional CNN classifier with a novel `NdLinear`-based model on the CIFAR-10 dataset. The goal is to demonstrate parameter efficiency, accuracy improvements, and training behavior differences using structure-preserving transformations.

---

## 🔧 Project Structure

```
.
├── main.py
├── models/
│   ├── baseline.py
│   └── ndlinear_model.py
├── utils/
│   ├── train.py
│   ├── visualize.py
│   └── report.py
├── results/
└── requirements.txt
```

---

## 🧠 Models

- **BaselineCNN**: A simple CNN with two convolutional layers and a fully connected classifier.
- **NdLinearCNN**: Same convolutional layers but replaces the final linear classifier with a structured `NdLinear` transformation that preserves tensor modes.

---

## 📊 Outputs

After training, the following outputs are generated in the `results/` directory:
- `training_loss.png` – Training loss over epochs
- `parameter_efficiency.png` – Accuracy per 1K parameters
- `ndlinear_report.md` – Markdown summary comparing performance

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/cifar10-ndlinear.git
cd cifar10-ndlinear
```

2. Set up your environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

---

## 🚀 Run the Training

```bash
python main.py
```

---

## 📈 Example Metrics (Expected)

| Model       | Params  | Test Accuracy | Efficiency (%/1K) |
|-------------|---------|----------------|--------------------|
| Baseline    | ~25,000 | 65.00%         | 2.60               |
| NdLinear    | ~11,000 | 68.00%         | 6.18               |

---

## 📚 Citation / Reference

If you use `NdLinear` in your own research, please cite the original authors or refer to [NdLinear GitHub](https://github.com/your-lib-url).

---

## 🧩 Todo / Ideas

- Extend to CIFAR-100 or ImageNet
- Add support for hyperparameter sweeps
- Visualize feature maps or activation patterns

---

## 🔗 License

MIT License. Use freely with attribution.
