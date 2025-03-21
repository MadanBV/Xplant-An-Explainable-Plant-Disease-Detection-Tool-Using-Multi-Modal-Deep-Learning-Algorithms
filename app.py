from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image # type: ignore
import os
import AI_model
from PIL import Image
import io
import base64
import uuid
import database_op
from datetime import datetime
from bson import ObjectId
import openai
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configuration for file uploads
app.config['Image_Uploaded'] = 'Image_Uploaded'
app.config['Gradcam_Uploaded'] = 'Gradcam_Result'
app.config['LIME_Uploaded'] = 'LIME_Result'
app.config['Plant_Uploaded'] = 'Plant_Result'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224) 

# Ensure directories exist
for folder in ['Image_Uploaded', 'Gradcam_Result', 'LIME_Result', 'Plant_Result']:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_data(image):
    
    img_io = io.BytesIO()
    image.save(img_io, format='png')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('ascii')

@app.route("/", methods=['GET', "POST"])
def index():
    
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    timestamp = datetime.now()

    success_message = None
    if name and email and message:
        try:
            database_op.save_contact(name, email, message, timestamp)
            success_message = "Your message has been saved. We'll get back to you soon!"
        except Exception as e:
            success_message = f"An error occurred: {str(e)}"
    elif request.method == "POST":
        success_message = "All fields are required."

    return render_template('index.html', contact_message=success_message)

@app.route("/user_dashboard", methods=['GET', "POST"])
def user_dashboard():
    
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    timestamp = datetime.now()

    success_message = None
    if name and email and message:
        try:
            database_op.save_contact(name, email, message, timestamp)
            success_message = "Your message has been saved. We'll get back to you soon!"
        except Exception as e:
            success_message = f"An error occurred: {str(e)}"
    elif request.method == "POST":
        success_message = "All fields are required."

    return render_template('user_dashboard.html', contact_message=success_message)

@app.route("/get_history_record/<int:index>")
def get_history_record(index):
    
    disease_data = database_op.disp_disease_data()
    if index < 0 or index >= len(disease_data):
        return jsonify({"error": "Index out of range"}), 400
    record = disease_data[index]

    user_img = image.load_img(record["Image uploaded"], target_size=IMG_SIZE)
    user_img = get_image_data(user_img)

    gradcam_img = image.load_img(record["gradcam image"], target_size=IMG_SIZE)
    gradcam_img = get_image_data(gradcam_img)

    lime_img = image.load_img(record["lime image"], target_size=IMG_SIZE)
    lime_img = get_image_data(lime_img)
    
    return jsonify({
        'Plant': record["plant"],
        'Disease': record["disease"],
        'Message': "Highlighted parts show the disease",
        'Image_uploaded': user_img,
        'gradcam_image': gradcam_img,
        'lime_image': lime_img
    })

@app.route("/research_dashboard", methods=["GET", "POST"])
def research_dashboard():
    
    disease_data = database_op.disp_disease_data()
    total_count = len(disease_data)
    return render_template(
        'research_dashboard.html',
        disease_data=disease_data,
        total_count=total_count,
    )

@app.route("/save_message", methods=["POST"])
def save_message():
    
    message = request.form.get("message")
    timestamp = datetime.now()

    if message:
        try:
            database_op.researcher_message(message, timestamp)
            flash("Message successfully sent to the developer!")
        except Exception as e:
            flash(f"Error saving message: {str(e)}")
    else:
        flash("Message cannot be empty!")

    return redirect(url_for("research_dashboard"))

@app.route("/upload_comment", methods=["POST"])
def upload_comment():
    
    record_id = request.form.get("record_id")
    user_comment = request.form.get("user_comment")
    developer_comment = request.form.get("developer_comment")

    if record_id and (user_comment or developer_comment):
        try:
            database_op.add_comments(record_id, user_comment, developer_comment)
            return jsonify({"success": True, "message": "Comment successfully saved!"}), 200
        except Exception as e:
            return jsonify({"success": False, "message": f"Error saving comments: {str(e)}"}), 500
    return jsonify({"success": False, "message": "Record ID and at least one comment are required!"}), 400

@app.route("/get_record/<int:index>")
def get_record(index):
    disease_data = database_op.disp_disease_data()
    if index < 0 or index >= len(disease_data):
        return jsonify({"error": "Index out of range"}), 400
    record = disease_data[index]
    user_img = image.load_img(record["Image uploaded"], target_size=IMG_SIZE)
    user_img = get_image_data(user_img)

    gradcam_img = image.load_img(record["gradcam image"], target_size=IMG_SIZE)
    gradcam_img = get_image_data(gradcam_img)

    lime_img = image.load_img(record["lime image"], target_size=IMG_SIZE)
    lime_img = get_image_data(lime_img)
    
    return jsonify({
        '_id': record["_id"],
        'Plant': record["plant"],
        'Disease': record["disease"],
        'Message': "Highlighted parts show the disease",
        'Image_uploaded': user_img,
        'gradcam_image': gradcam_img,
        'lime_image': lime_img
    })

@app.route("/get_user_message")
def get_user_message():
    
    try:
        user_messages = database_op.disp_user_message()
        return jsonify(user_messages)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch messages: {str(e)}"}), 500

@app.route("/uploads/<folder>/<filename>")
def uploaded_file(folder, filename):
    
    if folder in app.config:
        directory = app.config[folder]
        return send_from_directory(directory, filename)
    return jsonify({"error": "Invalid folder"}), 400

@app.route("/get_Research_message")
def get_Research_message():
    research_message = database_op.disp_research_message()
    return jsonify(research_message)

@app.route("/disease_detection", methods=["POST"])
def disease_detection():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("No selected file or invalid file type")
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['Image_Uploaded'], filename)
    file.save(file_path)

    # Perform AI model prediction
    results = AI_model.prediction(file_path)

    # Grad-CAM and Confidence Score
    gradcam_img, confidence_score = AI_model.plot_gradcam(file_path)
    gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
    gradcam_path = os.path.join(app.config['Gradcam_Uploaded'], gradcam_filename)
    gradcam_img.save(gradcam_path)
    gradcam_img_base64 = get_image_data(gradcam_img)

    # LIME Explanation
    lime_img = AI_model.explain_with_lime(file_path)
    lime_filename = f"lime_{uuid.uuid4().hex}.png"
    lime_path = os.path.join(app.config['LIME_Uploaded'], lime_filename)
    lime_img.save(lime_path)
    lime_img_base64 = get_image_data(lime_img)
    """
    shap_values = AI_model.explain_with_shap(file_path)
    mean_shap = np.abs(shap_values).mean()
    max_SHAP = np.abs(shap_values).max()
    """

    database_op.add_disease_data(results, file_path, gradcam_path, lime_path, datetime.now())

    return jsonify({
        'results': results,
        'Message': "Highlighted parts show the disease",
        'gradcam_img': gradcam_img_base64,
        'lime_img': lime_img_base64,
        'user_image': url_for('uploaded_file', folder='Image_Uploaded', filename=filename, _external=True),
        'confidence_score': round(confidence_score * 100, 2)
    })


@app.route("/plant_detection", methods=["POST"])
def plant_detection():
    if request.method == "POST":
        # Check if the file part is in the request
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Save the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['Plant_Uploaded'], filename)
            file.save(file_path)

            # Use the model to predict the plant type or health status
            Plant = AI_model.plant_predict(file_path)

            database_op.add_plant_data(Plant, file_path, datetime.now())

            # Return a JSON response
            return jsonify({
                'Plant': Plant,
            })        


@app.route("/developer_dashboard")
def developer_dashboard():
    return render_template('developer_dashboard.html')

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("user_message")

    if not user_message:
        return jsonify({"reply": "Please provide a message."})

    try:
        openai.api_key = "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"

        # Use gpt-3.5-turbo model
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Extract the reply
        reply = response['choices'][0]['message']['content'].strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)
