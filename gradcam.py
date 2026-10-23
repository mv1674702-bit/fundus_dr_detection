import tensorflow as tf

import numpy as np

import cv2

import os



# ============================================
# GENERATE GRAD-CAM HEATMAP
# ============================================

def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
):

    # Create model that gives:
    # convolution output + prediction

    grad_model = tf.keras.models.Model(

        inputs=model.inputs,

        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]

    )


    # Calculate gradients

    with tf.GradientTape() as tape:


        conv_outputs, predictions = grad_model(
            img_array
        )


        predicted_class = tf.argmax(
            predictions[0]
        )


        class_output = predictions[:, predicted_class]



    gradients = tape.gradient(

        class_output,

        conv_outputs

    )



    # Average gradients

    pooled_gradients = tf.reduce_mean(

        gradients,

        axis=(0,1,2)

    )



    conv_outputs = conv_outputs[0]



    # Weight feature maps

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]



    heatmap = tf.squeeze(
        heatmap
    )



    # Normalize

    heatmap = tf.maximum(

        heatmap,

        0

    )


    heatmap = heatmap / tf.reduce_max(

        heatmap

    )



    return heatmap.numpy()



# ============================================
# SAVE HEATMAP IMAGE
# ============================================

def save_heatmap(
        image_path,
        heatmap
):


    output_folder = "static/heatmaps"


    os.makedirs(

        output_folder,

        exist_ok=True

    )



    # Read original image

    img = cv2.imread(

        image_path

    )



    original = img.copy()



    # Resize heatmap

    heatmap = cv2.resize(

        heatmap,

        (img.shape[1], img.shape[0])

    )



    # Convert to 0-255

    heatmap = np.uint8(

        255 * heatmap

    )



    # Apply color map

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )



    # Combine original + heatmap

    superimposed_img = cv2.addWeighted(

        original,

        0.6,

        heatmap,

        0.4,

        0

    )



    filename = os.path.basename(

        image_path

    )


    output_path = os.path.join(

        output_folder,

        "gradcam_" + filename

    )



    cv2.imwrite(

        output_path,

        superimposed_img

    )



    return output_path