import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import preprocess

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "model/dr_model.h5"

TEST_CSV = "dataset/test.csv"

TEST_IMAGES_DIR = "dataset/test_images"

IMAGE_EXT = ".png"

CLASSES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


def main():

    model = load_model(MODEL_PATH)

    df = pd.read_csv(TEST_CSV)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():

        image_path = os.path.join(
            TEST_IMAGES_DIR,
            str(row["id_code"]) + IMAGE_EXT
        )

        if not os.path.exists(image_path):
            print("Skipping missing image:", image_path)
            continue

        processed = preprocess(image_path)

        prediction = model.predict(processed, verbose=0)

        y_pred.append(int(np.argmax(prediction)))
        y_true.append(int(row["diagnosis"]))

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
