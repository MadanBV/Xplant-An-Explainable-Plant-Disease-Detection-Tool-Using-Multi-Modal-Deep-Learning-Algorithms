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
Login_data = db['Login_data']

def add_disease_data(results, file_path, gradcam_path_Eff, gradcam_path_Vgg, gradcam_path_Mob, gradcam_path_Goo, lime_paths_dict, timestamp):
    try:
        prediction_data = {
            "results": results,
            "Image uploaded": file_path,
            "Eff gradcam image": gradcam_path_Eff,
            "Vgg gradcam image": gradcam_path_Vgg,
            "Mob gradcam image": gradcam_path_Mob,
            "Goo gradcam image": gradcam_path_Goo,
            "Eff lime image": lime_paths_dict.get("Eff_lime"),
            "Vgg lime image": lime_paths_dict.get("Vgg_lime"),
            "Mob lime image": lime_paths_dict.get("Mob_lime"),
            "Goo lime image": lime_paths_dict.get("Goo_lime"),
            "Res lime image": lime_paths_dict.get("Res_lime"),
            "Alex lime image": lime_paths_dict.get("Alex_lime"),
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
            "results": 1,
            "Image uploaded": 1,
            "Eff gradcam image": 1,
            "Vgg gradcam image": 1,
            "Mob gradcam image": 1,
            "Goo gradcam image": 1,
            "Eff lime image": 1,
            "Vgg lime image": 1,
            "Mob lime image": 1,
            "Goo lime image": 1,
            "Res lime image": 1,
            "Alex lime image": 1
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
    
def disp_plant_data():
    """Retrieve all plant data from MongoDB"""
    try:
        plant_data = list(Plant_data.find({}, {
            "_id": 1,
            "plant": 1,
            "Image uploaded": 1,
            "timestamp": 1
        }))

        for data in plant_data:
            data['_id'] = str(data['_id'])
        
        return plant_data
    except Exception as e:
        print(f"Error retrieving plant data: {e}")
        return []

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

def add_user(username, password, role):
    try:
        user_data = {"username": username, "password": password, "role": role}
        Login_data.insert_one(user_data)
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        return False

def validate_login(username, password, role):
    try:
        user = Login_data.find_one({"username": username, "password": password, "role": role})
        return user is not None
    except Exception as e:
        print(f"Error validating login: {e}")
        return False