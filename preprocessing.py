import cv2
import numpy as np
 
 
# ============================================
# IMAGE SIZE FOR EFFICIENTNET
# ============================================
 
IMG_SIZE = 224
 
 
# ============================================
# CORE PREPROCESSING (shared by train + predict)
# ------------------------------------------------
# IMPORTANT: this exact function is used both when
# training the model (via train.py's preprocessing_function)
# and when predicting (via preprocess() below). If the two
# pipelines ever differ, the model will perform poorly at
# inference time even though it trained/validated fine.
# ============================================
 
def preprocess_array(img):
    """
    Takes an image as a numpy array (H, W, 3) with values in
    range [0, 255], BGR or RGB (channel order does not matter
    since we immediately reduce to a single green-like channel),
    and returns a normalized (IMG_SIZE, IMG_SIZE, 3) float array
    in range [0, 1].
    """
 
    # --------------------------
    # Resize Image
    # --------------------------
 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
 
    # --------------------------
    # Extract Green Channel
    # (green channel shows the clearest contrast for
    # retinal blood vessels / lesions in fundus photography)
    # --------------------------
 
    green_channel = img[:, :, 1].astype(np.uint8)
 
    # --------------------------
    # CLAHE Enhancement
    # --------------------------
 
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
 
    enhanced = clahe.apply(green_channel)
 
    # --------------------------
    # Noise Reduction
    # --------------------------
 
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
 
    # --------------------------
    # NOTE: we deliberately do NOT divide by 255 here.
    # EfficientNetB0 (as provided by tf.keras.applications)
    # already contains its own internal Rescaling/Normalization
    # layer and expects raw pixel values in the [0, 255] range.
    # Normalizing here as well would double-rescale the image,
    # shrinking pixel values so much the model can't learn from
    # them (this is what caused near-random training accuracy).
    # --------------------------
 
    # --------------------------
    # Convert Gray to RGB
    # EfficientNet requires 3 channels
    # --------------------------
 
    rgb_image = np.stack(
        (blurred, blurred, blurred),
        axis=-1
    )
 
    return rgb_image.astype(np.float32)
 
 
# ============================================
# PREPROCESS FUNDUS IMAGE (single image, for prediction)
# ============================================
 
def preprocess(image_path):
 
    img = cv2.imread(image_path)
 
    if img is None:
        raise ValueError(
            "Image not found or invalid image path"
        )
 
    processed = preprocess_array(img)
 
    # Add Batch Dimension -> (1, 224, 224, 3)
    processed = np.expand_dims(processed, axis=0)
 
    return processed
 
 
# ============================================
# PREPROCESSING FUNCTION FOR KERAS ImageDataGenerator
# ------------------------------------------------
# Keras calls preprocessing_function with an image that has
# already been resized to target_size but is still in raw
# 0-255 pixel range and BEFORE any `rescale` is applied, so we
# do NOT set rescale=1./255 on the generator — normalization
# happens once, here, exactly as it does in preprocess() above.
# ============================================
 
def preprocess_for_training(img):
    return preprocess_array(img)
 