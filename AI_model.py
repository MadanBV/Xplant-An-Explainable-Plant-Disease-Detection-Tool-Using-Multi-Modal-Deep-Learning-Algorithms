import uuid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from scipy.ndimage import zoom
from PIL import Image
import cv2
import os
import pandas as pd
import numpy as np
import shap


IMG_SIZE = (224, 224)  

csv_file_path = "Diseases.csv"
class_df = pd.read_csv(csv_file_path)
Crop_list = [f"{row['Crop']}" for _, row in class_df.iterrows()]
Disease_list = [f"{row['Disease']}" for _, row in class_df.iterrows()]

#Load trained models
eff_model_path = r'Trained models\efficientnetv2S_10.keras'
eff_model = tf.keras.models.load_model(eff_model_path)

res_model_path = r'Trained models\final_resnet-10.keras'
res_model = tf.keras.models.load_model(res_model_path)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    return img_array

#CNN
def prediction(img_path):
    img = load_and_preprocess_image(img_path)

    eff_preds = eff_model.predict(img)
    eff_pred_class = np.argmax(eff_preds, axis=-1)[0]
    eff_max_value = np.max(eff_preds, axis=-1)[0]  

    res_preds = res_model.predict(img)
    res_pred_class = np.argmax(res_preds, axis=-1)[0]
    res_max_value = np.max(res_preds, axis=-1)[0]

    if eff_max_value > res_max_value:
        Plant = Crop_list[eff_pred_class]
        Disease = Disease_list[eff_pred_class]
        return Plant, Disease
    else:
        Plant = Crop_list[eff_pred_class]
        Disease = Disease_list[eff_pred_class]
        return Plant, Disease

#Plant detection
def plant_predict(img_path):

    csv_file_path = "Plants.csv"
    plant_df = pd.read_csv(csv_file_path)
    Plants = [f"{row['Plants']}" for _, row in plant_df.iterrows()]

    Plant_model_path = r'Trained models\Plant_detect_EfficientNetV2S.keras'
    plant_model = tf.keras.models.load_model(Plant_model_path)

    img = load_and_preprocess_image(img_path)

    preds = plant_model.predict(img)
    pred_class = np.argmax(preds, axis=-1)[0] 

    Plant = Plants[pred_class]
    return Plant

#LIME
def perturbation_visualization(image, mask):
    # Resize mask to match image dimensions
    mask_resized = zoom(mask, (image.size[1] / mask.shape[0], image.size[0] / mask.shape[1]), order=1)
    mask_resized = np.clip(mask_resized, 0, 1)  

    # Create a gray image
    gray_color = (128, 128, 128)  
    gray_image = Image.new('RGB', image.size, gray_color)

    # Create the overlay
    image_array = np.array(image)
    gray_array = np.array(gray_image)
    mask_3d = np.expand_dims(mask_resized, axis=2)
    overlay = (image_array * mask_3d + gray_array * (1 - mask_3d)).astype(np.uint8)

    return Image.fromarray(overlay)

def explain_with_lime(img_path):
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    explainer = lime_image.LimeImageExplainer()
    
    explanation = explainer.explain_instance(
        img_array.astype('double'),
        eff_model.predict,
        top_labels=3,
        hide_color=None, 
        num_samples=1000
    )
    
    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = perturbation_visualization(img, mask)
    return lime_img  


#GRADCAM
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    
    guided_grads = grads * conv_outputs
    heatmap = tf.reduce_sum(guided_grads, axis=-1)
    
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + tf.keras.backend.epsilon())
    return heatmap

def plot_gradcam(img_path, last_conv_layer_name="top_conv"):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = load_and_preprocess_image(img_path)
    
    heatmap = get_gradcam_heatmap(eff_model, img_array, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    # Get the model's confidence score
    preds = eff_model.predict(img_array)
    confidence_score = np.max(preds)  # Extract highest probability

    return Image.fromarray(superimposed_img[..., ::-1]), confidence_score


