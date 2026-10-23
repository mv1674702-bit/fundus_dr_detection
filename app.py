# ==========================
# IMPORT LIBRARIES
# ==========================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    flash,
    send_file
)

import os
import sqlite3
import numpy as np

from tensorflow.keras.models import load_model

# Local Files
from preprocessing import preprocess
from database import (
    create_database,
    create_users_table,
    register_user,
    login_user,
    insert_patient,
    get_all_patients,
    search_patient
)

from gradcam import (
    make_gradcam_heatmap,
    save_heatmap
)

from report import generate_report


# ==========================
# FLASK CONFIGURATION
# ==========================

app = Flask(__name__)

# In production set this via an environment variable instead of
# hardcoding it, e.g.: app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fundus_dr_secret_key")

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "static/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ==========================
# CREATE FOLDERS
# ==========================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ==========================
# DATABASE INITIALIZATION
# ==========================

create_database()
create_users_table()


# ==========================
# LOAD MODEL
# ==========================

MODEL_PATH = "model/dr_model.h5"

model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print(
        "WARNING: No trained model found at "
        + MODEL_PATH
        + ". Train one with train.py first — "
        + "the app will run, but /predict will show an error until a "
        + "model file exists."
    )


# ==========================
# CLASS LABELS
# ==========================

classes = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Last conv layer of EfficientNetB0, used for Grad-CAM
LAST_CONV_LAYER = "top_conv"


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template("index.html", username=session["user"])


# ==========================
# REGISTER
# ==========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        try:
            register_user(username, email, password)
            flash("Registration Successful", "success")
            return redirect("/login")

        except sqlite3.IntegrityError:
            flash("User Already Exists", "danger")

    return render_template("register.html")


# ==========================
# LOGIN
# ==========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = login_user(email, password)

        if user:
            session["user"] = user[1]
            flash("Login Successful", "success")
            return redirect("/")
        else:
            flash("Invalid Credentials", "danger")

    return render_template("login.html")


# ==========================
# LOGOUT
# ==========================

@app.route("/logout")
def logout():

    session.clear()
    flash("Logged Out Successfully", "info")
    return redirect("/login")


# ============================================
# PREDICT DIABETIC RETINOPATHY
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    if model is None:
        flash(
            "No trained model found. Run train.py to create "
            "model/dr_model.h5 before predicting.",
            "danger"
        )
        return redirect("/")

    # --------------------------
    # Get Form Data
    # --------------------------

    patient = request.form.get("patient", "").strip()
    age = request.form.get("age", "").strip()
    image = request.files.get("image")

    if not patient or not age:
        flash("Please Provide Patient Name And Age", "warning")
        return redirect("/")

    if image is None or image.filename == "":
        flash("Please Select Image", "warning")
        return redirect("/")

    if not allowed_file(image.filename):
        flash("Only PNG/JPG/JPEG images are supported", "warning")
        return redirect("/")

    # --------------------------
    # Save Uploaded Image
    # --------------------------

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(image_path)

    # --------------------------
    # Image Preprocessing
    # --------------------------

    processed_image = preprocess(image_path)

    # --------------------------
    # Model Prediction
    # --------------------------

    prediction = model.predict(processed_image)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    disease = classes[class_index]

    # --------------------------
    # AI Observation
    # --------------------------

    remarks = {
        "No DR":
            "No signs of diabetic retinopathy detected. Continue routine eye examination.",
        "Mild":
            "Mild diabetic retinopathy detected. Regular follow-up is recommended.",
        "Moderate":
            "Moderate diabetic retinopathy detected. Consult an ophthalmologist.",
        "Severe":
            "Severe diabetic retinopathy detected. Immediate medical attention is advised.",
        "Proliferative DR":
            "High risk diabetic retinopathy detected. Urgent specialist consultation is recommended."
    }

    observation = remarks[disease]

    # --------------------------
    # Generate Grad-CAM
    # --------------------------

    heatmap_path = None

    try:
        heatmap = make_gradcam_heatmap(
            processed_image,
            model,
            last_conv_layer_name=LAST_CONV_LAYER
        )

        heatmap_path = save_heatmap(image_path, heatmap)

    except Exception as exc:
        print("Grad-CAM generation failed:", exc)

    # --------------------------
    # Generate PDF Report
    # --------------------------

    report_path = generate_report(
        patient,
        age,
        disease,
        round(confidence, 2)
    )

    # --------------------------
    # Save Patient Record
    # --------------------------

    insert_patient(
        patient,
        age,
        disease,
        round(confidence, 2),
        image_path
    )

    # --------------------------
    # Display Result
    # --------------------------

    return render_template(
        "result.html",
        patient=patient,
        age=age,
        result=disease,
        confidence=round(confidence, 2),
        observation=observation,
        image=image_path,
        heatmap=heatmap_path,
        report=report_path
    )


# ============================================
# DOWNLOAD PDF REPORT
# ============================================

@app.route("/download_report/<path:filename>")
def download_report(filename):
    return send_file(filename, as_attachment=True)


# ============================================
# PATIENT HISTORY
# ============================================

@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    patients = get_all_patients()

    return render_template("history.html", patients=patients)


# ============================================
# SEARCH PATIENT
# ============================================

@app.route("/search", methods=["POST"])
def search():

    if "user" not in session:
        return redirect("/login")

    name = request.form.get("name", "")

    patients = search_patient(name)

    return render_template("history.html", patients=patients)


# ============================================
# DOCTOR DASHBOARD
# ============================================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    patients = get_all_patients()

    total_patients = len(patients)

    dr_count = {
        "No DR": 0,
        "Mild": 0,
        "Moderate": 0,
        "Severe": 0,
        "Proliferative DR": 0
    }

    for patient in patients:
        disease = patient[3]
        if disease in dr_count:
            dr_count[disease] += 1

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        dr_count=dr_count,
        recent=patients[:5]
    )


# ============================================
# EXPORT PATIENT DATA
# ============================================

@app.route("/export")
def export():

    if "user" not in session:
        return redirect("/login")

    import pandas as pd

    patients = get_all_patients()

    df = pd.DataFrame(
        patients,
        columns=[
            "ID",
            "Name",
            "Age",
            "Prediction",
            "Confidence",
            "Image",
            "Date"
        ]
    )

    file_path = "static/patient_records.csv"

    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)


# ============================================
# ERROR HANDLING
# ============================================

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
