import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # type: ignore
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

mob_model_path = r'Trained models\MobileNetV2.keras'
mob_model = tf.keras.models.load_model(mob_model_path)

Google_model_path = r'Trained models\GoogleNetV4.keras'
google_model = tf.keras.models.load_model(Google_model_path)

alex_model_path = r'Trained models\AlexNet.keras'
alex_model = tf.keras.models.load_model(alex_model_path)

VGG_model_path = r'Trained models\VGG19.keras'
VGG_model = tf.keras.models.load_model(VGG_model_path)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    return img_array

def load_and_preprocess_google_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    return img_array

#CNN
def predict_model(model, img, crop_list, disease_list):
    preds = model.predict(img)
    pred_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    return pred_class, confidence, crop_list[pred_class], disease_list[pred_class]


def prediction(img_path):
    img = load_and_preprocess_image(img_path)
    google_img = load_and_preprocess_google_image(img_path)

    model_info = [
        ("VGG19", VGG_model, img),
        ("GoogleNetV4", google_model, google_img),
        ("AlexNet", alex_model, img),
        ("EfficientNetV2", eff_model, img),
        ("ResNet152V2", res_model, img),
        ("MobileNetV2", mob_model, img)
    ]

    results = []
    for name, model, image_input in model_info:
        cls_idx, confidence, crop, disease = predict_model(model, image_input, Crop_list, Disease_list)
        results.append({
            "Model": name,
            "Plant": crop,
            "Disease": disease,
            "Confidence": confidence
        })
    return results

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

def explain_with_lime(img_path, model, preprocess_func):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_array.astype('double'),
        model.predict,
        top_labels=1,
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


def plot_Eff_gradcam(img_path, last_conv_layer_name="top_conv"):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = load_and_preprocess_image(img_path)
    
    heatmap = get_gradcam_heatmap(eff_model, img_array, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    return Image.fromarray(superimposed_img[..., ::-1])


def plot_Vgg_gradcam(img_path, last_conv_layer_name="block5_conv4"):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = load_and_preprocess_image(img_path)

    heatmap = get_gradcam_heatmap(VGG_model, img_array, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img[..., ::-1])


def plot_mob_gradcam(img_path, last_conv_layer_name="Conv_1"):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = load_and_preprocess_image(img_path)

    heatmap = get_gradcam_heatmap(mob_model, img_array, last_conv_layer_name="Conv_1")
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img[..., ::-1])

def plot_Goo_gradcam(img_path, last_conv_layer_name="mixed10"):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = load_and_preprocess_google_image(img_path)

    heatmap = get_gradcam_heatmap(google_model, img_array, last_conv_layer_name="mixed10")
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img[..., ::-1])