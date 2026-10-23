import pandas as pd
import tensorflow as tf
 
# Read CSV files
train_df = pd.read_csv("dataset/train_1.csv")
valid_df = pd.read_csv("dataset/valid.csv")
 
# Add .png extension
train_df["id_code"] = train_df["id_code"].astype(str) + ".png"
valid_df["id_code"] = valid_df["id_code"].astype(str) + ".png"
 
# Convert labels to string
train_df["diagnosis"] = train_df["diagnosis"].astype(str)
valid_df["diagnosis"] = valid_df["diagnosis"].astype(str)
 
# Image generators
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
 
# Training dataset
train_data = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory="dataset/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)
 
# Validation dataset
valid_data = valid_gen.flow_from_dataframe(
    dataframe=valid_df,
    directory="dataset/val_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
 
# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
 
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
 
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
 
    tf.keras.layers.Flatten(),
 
    tf.keras.layers.Dense(128, activation="relu"),
 
    tf.keras.layers.Dense(5, activation="softmax")
])
 
# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
 
# Print dataset information
print("Training images:", train_data.samples)
print("Validation images:", valid_data.samples)
 
# Train model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)
 
# Save model
model.save("model/dr_model.h5")
print("Model saved successfully!")
 