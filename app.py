from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import AI_model
from PIL import Image
import io
import base64
import uuid
import database_op
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['Image_Uploaded'] = 'Image_Uploaded'
app.config['Gradcam_Uploaded'] = 'Gradcam_Result'
app.config['LIME_Uploaded'] = 'LIME_Result'
app.config['Plant_Uploaded'] = 'Plant_Result'

def get_image_data(image):
    img_io = io.BytesIO()
    image.save(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
    return img_base64

@app.route("/", methods=['GET', "POST"])
def index():

    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    timestamp = datetime.now()

    if name and email and message:
        try:
            database_op.save_contact(name, email, timestamp, message)
            success_message = "Your message has been saved. We'll get back to you soon!"
        except Exception as e:
            success_message = f"An error occurred: {str(e)}"
    else:
        success_message = "All fields are required."

    return render_template('index.html', contact_message=success_message)

@app.route("/user_dashboard")
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route("/research_dashboard")
def research_dashboard():
    return render_template('research_dashboard.html')

@app.route("/get_disease_data", methods=["GET", "POST"])
def get_disease_data():
    try:
        data = database_op.disp_disease_data()
        print("success")
        return jsonify(data)
    except Exception as e:
        print("error")
        return jsonify({"error": str(e)}), 500

@app.route("/developer_dashboard")
def developer_dashboard():
    return render_template('developer_dashboard.html')

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

            # Return a JSON response
            return jsonify({
                'Plant': Plant,
            })        

    #return render_template('plant_detection.html')


@app.route("/disease_detection", methods=["POST"])
def disease_detection():

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['Image_Uploaded'], filename)
            file.save(file_path)

            # Perform prediction
            Plant, Disease = AI_model.prediction(file_path)

            if Disease == "healthy":

                timestamp = datetime.now()
                database_op.add_disease_data(Plant, Disease, file_path, None, None, timestamp)

                return jsonify({
                    'Plant': Plant,
                    'Message': "Plant don't have any disease"
                })
            
            else:
                # Generate Grad-CAM
                gradcam_img = AI_model.plot_gradcam(file_path)
                gradcam_img_path = os.path.join(app.config['Gradcam_Uploaded'], f"gradcam_{uuid.uuid4().hex}.png")
                gradcam_img.save(gradcam_img_path)
                gradcam_img_base64 = get_image_data(gradcam_img)

                # Generate LIME explanation
                lime_img = AI_model.explain_with_lime(file_path)
                lime_img_path = os.path.join(app.config['LIME_Uploaded'], f"lime_{uuid.uuid4().hex}.png")
                lime_img.save(lime_img_path)
                lime_img_base64 = get_image_data(lime_img)

                timestamp = datetime.now()
                database_op.add_disease_data(Plant, Disease, file_path, gradcam_img_path, lime_img_path, timestamp)

                # Return results as JSON to display in popup
                return jsonify({
                    'Plant': Plant,
                    'Disease': Disease,
                    'Mwssage': "Highlighted part below show disease",
                    'gradcam_img': gradcam_img_base64,
                    'lime_img': lime_img_base64
                })
    #return render_template('disease_detection.html')

if __name__ == "__main__":
    app.run(debug=True)