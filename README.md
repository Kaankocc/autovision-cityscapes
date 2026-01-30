# Autovision Cityscapes

Semantic segmentation on Cityscapes-style data using a UNet model. This repo keeps data local (empty in the repo), code modular, and experiments reproducible via notebooks.

## Project structure

```
autovision-cityscapes/
├── data/               # Empty locally (add your datasets here; see .gitignore)
├── models/             # Best .pth checkpoints
├── notebooks/          # Step-by-step experiment record
│   ├── 01_EDA.ipynb
│   ├── 02_Training.ipynb
│   └── 03_Validation.ipynb
├── src/                # Modular reusable Python scripts
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Place your Cityscapes (or Kaggle Cityscapes) data under `data/` (e.g. `data/raw/cityscapes_data/`). The folder is gitignored so it stays local.

3. (Optional) Preprocess masks with:

   ```bash
   python -m src.utils
   ```

## Usage

- **EDA:** Open `notebooks/01_EDA.ipynb` to explore the dataset.
- **Training:** Use `notebooks/02_Training.ipynb` to train the UNet; best weights go in `models/`.
- **Validation:** Use `notebooks/03_Validation.ipynb` to evaluate and visualize predictions.

## Dependencies

See `requirements.txt` for main libraries (e.g. `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`).
