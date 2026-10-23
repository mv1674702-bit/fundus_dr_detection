import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/dr_model.h5")

# Class labels
classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Load image
img = image.load_img("dataset/train_images/3ddb86eb530e.png", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Prediction:", classes[predicted_class])
print("Confidence:", np.max(prediction) * 100, "%")