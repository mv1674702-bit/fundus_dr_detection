# 🩺 Fundus Image Based Diabetic Retinopathy Detection Using Deep Learning

## 📌 Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can cause vision loss if not detected early.

This project is an AI-based retinal image analysis system that detects the severity level of Diabetic Retinopathy from fundus images using Deep Learning and Image Processing techniques.

The system uses EfficientNetB0 with transfer learning for classification and Grad-CAM visualization to explain the regions of the retina that influenced the prediction.

---

## 🚀 Features

### 👁️ Fundus Image Analysis
- Upload retinal fundus images
- Automated image preprocessing (green-channel extraction + CLAHE contrast enhancement)
- AI-based DR severity classification

### 🧠 Deep Learning Prediction
The model classifies images into five categories:
- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

### 🔥 Grad-CAM Visualization
- Visual explanation of what the model looked at
- Highlights the retinal regions that most influenced the prediction
- Improves interpretability for clinicians

### 📄 Medical Report Generation
Automatically generates a PDF report containing:
- Patient details
- DR prediction
- Confidence score
- AI observation
- Date and time

### 👨‍⚕️ Doctor Dashboard
- Patient history
- Search patient records
- Prediction statistics by severity
- Export patient data to CSV

### 🔐 Authentication
- Registration
- Login
- Session-based access control

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, SQLite |
| Machine Learning | TensorFlow / Keras, EfficientNetB0 (transfer learning) |
| Image Processing | OpenCV, CLAHE |
| Frontend | HTML, CSS (custom design system) |
| Report Generation | ReportLab |

---

## 📂 Project Structure

```
fundus_dr_detection/
├── app.py                  # Flask application & routes
├── database.py             # SQLite helpers (users + patients)
├── gradcam.py              # Grad-CAM heatmap generation
├── predict.py              # Standalone CLI prediction script
├── preprocessing.py        # Shared image preprocessing (train + inference)
├── report.py               # PDF report generation
├── train.py                # Model training script
├── evaluate.py             # Evaluate a trained model on test.csv
├── requirements.txt
├── .gitignore
├── model/
│   └── dr_model.h5         # Trained model goes here (not included)
├── dataset/
│   ├── train_1.csv         # id_code, diagnosis (provided)
│   ├── valid.csv           # id_code, diagnosis (provided)
│   ├── test.csv            # id_code, diagnosis (provided)
│   ├── train_images/       # you add: <id_code>.png per row in train_1.csv
│   ├── val_images/         # you add: <id_code>.png per row in valid.csv
│   └── test_images/        # you add: <id_code>.png per row in test.csv
├── static/
│   ├── css/style.css
│   ├── uploads/            # uploaded fundus images land here
│   ├── heatmaps/           # generated Grad-CAM images
│   └── reports/            # generated PDF reports
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── index.html
    ├── result.html
    ├── history.html
    ├── dashboard.html
    └── 404.html
```

---

## 📊 About the dataset CSVs

`train_1.csv`, `valid.csv`, and `test.csv` follow the APTOS-2019 style format:

```
id_code,diagnosis
000c1434d8d7,2
```

- `id_code` — the image filename without extension
- `diagnosis` — an integer 0–4 mapping to `No DR, Mild, Moderate, Severe, Proliferative DR`

These CSVs only contain **labels** — the actual `.png`/`.jpg` fundus photographs are not included in this bundle (they're large binary files). Place them in:

- `dataset/train_images/<id_code>.png` for rows in `train_1.csv`
- `dataset/val_images/<id_code>.png` for rows in `valid.csv`
- `dataset/test_images/<id_code>.png` for rows in `test.csv`

If your images are `.jpg`, change `IMAGE_EXT` at the top of `train.py` / `evaluate.py`.

---

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🏋️ Train the model

1. Add your fundus images under `dataset/train_images/`, `dataset/val_images/` (see above).
2. Run:

```bash
python train.py
```

This trains an EfficientNetB0 classifier head first with the base network frozen, then fine-tunes the top layers of the base network at a low learning rate. The best checkpoint (by validation accuracy) is saved to `model/dr_model.h5`.

**Important:** `train.py` and `preprocessing.py` share the exact same `preprocess_array()` function (green-channel extraction, CLAHE, normalization). This keeps training and inference consistent — if you change one, change both, or the deployed model's accuracy won't match what you saw during training.

## 📈 Evaluate the model

Once you have `model/dr_model.h5` and images in `dataset/test_images/`:

```bash
python evaluate.py
```

Prints a classification report and confusion matrix against `test.csv`.

## ▶️ Run the web app

```bash
python app.py
```

Visit `http://localhost:5000`, register an account, log in, and upload a fundus image to screen it.

> The app starts even without a trained model, but `/predict` will show a message asking you to run `train.py` first.

## 🔍 Command-line prediction

```bash
python predict.py
```

Edit `image_path` at the bottom of `predict.py`, or import `predict_image()` into your own script.

---

## ⚠️ Disclaimer

This project is an educational / assistive screening tool. It is **not** a substitute for professional medical diagnosis. All predictions should be reviewed by a qualified ophthalmologist.
