from pymongo import MongoClient
from datetime import datetime
import os

client = MongoClient('mongodb://localhost:27017/')
db = client['Plant_disease']
Disease_data = db['Diseases']
User_message_data = db['User_message']

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

def disp_disease_data():
    Disp_data = Disease_data.find({}, {"_id": 0, "plant": 1, "disease": 1, "file_path": 1, "gradcam_path": 1, "lime_path": 1})
    #total_count = Disease_data.count_documents()
    disp_data = list(Disp_data)
    return(disp_data)

def save_contact(name, email, message, timestamp):
    message_data = {
        "name": name,
        "email": email,
        "message": message,
        "timestamp": timestamp
    }
    User_message_data.insert_one(message_data)