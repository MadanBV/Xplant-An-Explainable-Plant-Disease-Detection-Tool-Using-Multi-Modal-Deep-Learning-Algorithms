from pymongo import MongoClient
from datetime import datetime
import os

client = MongoClient('mongodb://localhost:27017/')
db = client['Plant_disease']
Disease_data = db['Diseases']

def add_disease_data(plant, disease, file_path, gradcam_path, lime_path, timestamp):
    """Save prediction data to MongoDB"""
    prediction_data = {
        "plant": plant,
        "disease": disease,
        "Image uploaded": file_path,
        "gradcam image": gradcam_path,
        "lime image": lime_path,
        "timestamp": timestamp
    }
    Disease_data.insert_one(prediction_data)