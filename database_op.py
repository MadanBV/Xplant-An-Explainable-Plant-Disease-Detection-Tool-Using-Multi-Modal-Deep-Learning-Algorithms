from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

# MongoDB Client and Database Initialization
client = MongoClient('mongodb://localhost:27017/')
db = client['Plant_disease']
Disease_data = db['Diseases']
Plant_data = db['Plants']
User_message_data = db['User_message']
Researcher_message = db['Research_message']

def add_disease_data(results, file_path, gradcam_path, lime_path, timestamp):
    try:
        prediction_data = {
            "results": results,
            "Image uploaded": file_path,
            "gradcam image": gradcam_path,
            "lime image": lime_path,
            "timestamp": timestamp,
            "user_comment": "",
            "developer_comment": ""
        }
        result = Disease_data.insert_one(prediction_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error inserting disease data: {e}")
        return None

def disp_disease_data():
    """Retrieve all disease data from MongoDB"""
    try:
        Disp_data = list(Disease_data.find({}, {
            "_id": 1,
            "plant": 1,
            "disease": 1,
            "Image uploaded": 1,
            "gradcam image": 1,
            "lime image": 1
        }))

        for data in Disp_data:
            data['_id'] = str(data['_id'])
        
        return Disp_data
    except Exception as e:
        print(f"Error retrieving disease data: {e}")
        return []
    
def add_plant_data(plant, file_path, timestamp):
    """Save prediction data to MongoDB"""
    try:
        prediction_data = {
            "plant": plant,
            "Image uploaded": file_path,
            "timestamp": timestamp,
            "comments": []  # Initialize empty comments list
        }
        result = Plant_data.insert_one(prediction_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error inserting disease data: {e}")
        return None

def save_contact(name, email, message, timestamp):
    """Save user contact/message data to MongoDB"""
    try:
        message_data = {
            "name": name,
            "email": email,
            "message": message,
            "timestamp": timestamp
        }
        result = User_message_data.insert_one(message_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving contact: {e}")
        return None

def disp_user_message():
    """Retrieve all user messages from MongoDB"""
    try:
        disp_msg = list(User_message_data.find({}, {
            "_id": 1,
            "name": 1,
            "email": 1,
            "message": 1,
            "timestamp": 1
        }))
        for msg in disp_msg:
            msg['_id'] = str(msg['_id'])  # Convert ObjectId to string
        return disp_msg
    except Exception as e:
        print(f"Error retrieving user messages: {e}")
        return []

def researcher_message(message, timestamp):
    """Save researcher message to MongoDB"""
    try:
        message_data = {
            "message": message,
            "timestamp": timestamp
        }
        result = Researcher_message.insert_one(message_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving researcher message: {e}")
        return None
    
def disp_research_message():
    """Retrieve all user messages from MongoDB"""
    try:
        disp_msg = list(User_message_data.find({}, {
            "_id": 1,
            "message": 1,
            "timestamp": 1
        }))
        for msg in disp_msg:
            msg['_id'] = str(msg['_id'])  # Convert ObjectId to string
        return disp_msg
    except Exception as e:
        print(f"Error retrieving user messages: {e}")
        return []

def add_comments(record_id, user_comment, developer_comment):
    """Add user and developer comments to a specific disease record"""
    try:
        result = Disease_data.update_one(
            {"_id": ObjectId(record_id)},
            {"$set": {
                "user_comment": user_comment,
                "developer_comment": developer_comment,
            }}
        )
        if result.modified_count > 0:
            return True
        return False
    except Exception as e:
        print(f"Error adding comments: {e}")
        return False
