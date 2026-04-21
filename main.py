"""
Health SymptomSense - Personalized Medical Recommendation System
Flask Backend with MongoDB, JWT Authentication, and ML Disease Prediction
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_cors import CORS
from functools import wraps
import numpy as np
import pandas as pd
import pickle
import os
import json
import hashlib
import secrets
import datetime
from dotenv import load_dotenv
from bson import ObjectId
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# ============================================================
# Flask App Configuration
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-2026')
CORS(app)

# ============================================================
# MongoDB Configuration
# ============================================================
try:
    from pymongo import MongoClient
    MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    uri_db_name = urlparse(MONGO_URI).path.lstrip('/')
    MONGO_DB = os.getenv('MONGODB_DB_NAME') or (uri_db_name if uri_db_name else 'health_symptomsense')
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    client.server_info()
    db = client[MONGO_DB]
    users_collection = db['users']
    predictions_collection = db['predictions']
    contacts_collection = db['contacts']
    # Create indexes
    users_collection.create_index('email', unique=True)
    MONGO_AVAILABLE = True
    print("[OK] MongoDB connected successfully!")
except Exception as e:
    MONGO_AVAILABLE = False
    print(f"[WARN] MongoDB not available: {e}")
    print("       Running in file-based fallback mode.")
    # Fallback: in-memory storage
    _users_store = {}
    _predictions_store = []
    _contacts_store = []

# ============================================================
# JWT Helper Functions
# ============================================================
JWT_SECRET = os.getenv('JWT_SECRET_KEY', 'jwt-dev-secret-2026')
JWT_EXPIRY = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 86400))

import base64
import hmac
import time


def create_jwt_token(user_id, email, name):
    """Create a simple JWT-like token"""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode()
    payload_data = {
        "user_id": str(user_id),
        "email": email,
        "name": name,
        "exp": int(time.time()) + JWT_EXPIRY,
        "iat": int(time.time())
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode()
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{signature}"


def verify_jwt_token(token):
    """Verify and decode a JWT token"""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        expected_sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload_data = json.loads(base64.urlsafe_b64decode(payload + '=='))
        if payload_data.get('exp', 0) < int(time.time()):
            return None
        return payload_data
    except Exception:
        return None


def hash_password(password):
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password, stored_hash):
    """Verify password against stored hash"""
    try:
        salt, hashed = stored_hash.split(':')
        expected = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(hashed, expected.hex())
    except Exception:
        return False


def token_required(f):
    """Decorator for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        # Check session
        if not token and 'auth_token' in session:
            token = session['auth_token']
        if not token:
            return jsonify({"error": "Authentication required", "authenticated": False}), 401
        user_data = verify_jwt_token(token)
        if not user_data:
            return jsonify({"error": "Invalid or expired token", "authenticated": False}), 401
        return f(user_data, *args, **kwargs)
    return decorated


# ============================================================
# Load ML Datasets
# ============================================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
severity = pd.read_csv("datasets/Symptom-severity.csv")

# ============================================================
# Load ML Model
# ============================================================
model_paths = ["models/svc.pkl", "models/svc .pkl"]
svc = None
for model_path in model_paths:
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            svc = pickle.load(model_file)
        break
if svc is None:
    raise FileNotFoundError(
        "No SVM model found. Run `python retrain.py` to generate models/svc.pkl."
    )

# ============================================================
# Symptoms and Diseases Dictionaries
# ============================================================
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# ============================================================
# Helper Functions  
# ============================================================
def helper(dis):
    """Get all information about a disease"""
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


def get_predicted_value(patient_symptoms):
    """Predict disease from symptoms using SVM model"""
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    predicted_idx = int(svc.predict([input_vector])[0])
    confidence = None
    if hasattr(svc, "predict_proba"):
        proba = svc.predict_proba([input_vector])[0]
        confidence = float(np.max(proba) * 100.0)
    return diseases_list[predicted_idx], confidence


def get_severity_score(symptom_list):
    """Calculate severity score from symptoms"""
    total = 0
    count = 0
    for s in symptom_list:
        match = severity[severity['Symptom'].str.strip().str.lower() == s.strip().lower()]
        if not match.empty:
            total += match['weight'].values[0]
            count += 1
    return round((total / max(count, 1)) * 15, 1)  # Scale to ~100


# ============================================================
# Page Routes (Template Rendering)
# ============================================================
@app.route("/")
def index():
    user = None
    if 'auth_token' in session:
        user_data = verify_jwt_token(session['auth_token'])
        if user_data:
            user = user_data
    return render_template("index.html", user=user)


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/developer')
def developer():
    return render_template("developer.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")


# ============================================================
# Authentication API Endpoints
# ============================================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json() if request.is_json else request.form
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not all([name, email, password]):
        return jsonify({"error": "All fields are required", "success": False}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters", "success": False}), 400
    
    hashed_pw = hash_password(password)
    
    if MONGO_AVAILABLE:
        # Check if user exists
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered", "success": False}), 409
        
        user_doc = {
            "name": name,
            "email": email,
            "password": hashed_pw,
            "avatar": name[0].upper(),
            "created_at": datetime.datetime.utcnow(),
            "predictions_count": 0
        }
        result = users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)
    else:
        if email in _users_store:
            return jsonify({"error": "Email already registered", "success": False}), 409
        user_id = secrets.token_hex(12)
        _users_store[email] = {
            "id": user_id, "name": name, "email": email,
            "password": hashed_pw, "avatar": name[0].upper(),
            "predictions_count": 0
        }
    
    token = create_jwt_token(user_id, email, name)
    session['auth_token'] = token
    
    return jsonify({
        "success": True,
        "message": "Registration successful",
        "token": token,
        "user": {"id": user_id, "name": name, "email": email, "avatar": name[0].upper()}
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json() if request.is_json else request.form
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not all([email, password]):
        return jsonify({"error": "Email and password are required", "success": False}), 400
    
    if MONGO_AVAILABLE:
        user = users_collection.find_one({"email": email})
        if not user or not verify_password(password, user['password']):
            return jsonify({"error": "Invalid email or password", "success": False}), 401
        user_id = str(user['_id'])
        name = user['name']
        avatar = user.get('avatar', name[0].upper())
        pred_count = user.get('predictions_count', 0)
    else:
        user = _users_store.get(email)
        if not user or not verify_password(password, user['password']):
            return jsonify({"error": "Invalid email or password", "success": False}), 401
        user_id = user['id']
        name = user['name']
        avatar = user.get('avatar', name[0].upper())
        pred_count = user.get('predictions_count', 0)
    
    token = create_jwt_token(user_id, email, name)
    session['auth_token'] = token
    
    return jsonify({
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {"id": user_id, "name": name, "email": email, "avatar": avatar, "predictions_count": pred_count}
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.pop('auth_token', None)
    return jsonify({"success": True, "message": "Logged out successfully"}), 200


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(user_data):
    """Get current authenticated user info"""
    return jsonify({
        "success": True,
        "user": {
            "id": user_data['user_id'],
            "name": user_data['name'],
            "email": user_data['email'],
            "avatar": user_data['name'][0].upper()
        }
    }), 200


# ============================================================
# Disease Prediction API Endpoints
# ============================================================
@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Get list of all available symptoms with metadata"""
    symptom_list = []
    for symptom, idx in sorted(symptoms_dict.items(), key=lambda x: x[1]):
        display_name = symptom.replace('_', ' ').title()
        # Get severity weight
        match = severity[severity['Symptom'].str.strip().str.lower() == symptom.replace('_', ' ').strip().lower()]
        weight = int(match['weight'].values[0]) if not match.empty else 3
        symptom_list.append({
            "id": idx,
            "key": symptom,
            "name": display_name,
            "severity": weight
        })
    return jsonify({"success": True, "symptoms": symptom_list, "total": len(symptom_list)}), 200


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict disease from symptoms
    
    Meta Input:
    {
        "symptoms": ["itching", "skin_rash", "fatigue"],
        "patient_info": {
            "age": 25,
            "gender": "male"  
        }
    }
    
    Meta Output:
    {
        "success": true,
        "prediction": {
            "disease": "Fungal infection",
            "confidence": 85.0,
            "severity_score": 45.2,
            "description": "...",
            "precautions": [...],
            "medications": [...],
            "diets": [...],
            "workouts": [...]
        },
        "meta": {
            "model": "SVM (RBF Kernel)",
            "symptoms_count": 3,
            "timestamp": "..."
        }
    }
    """
    data = request.get_json() if request.is_json else None
    
    if not data:
        # Fallback to form data
        symptoms_str = request.form.get('symptoms', '')
        symptom_list = [s.strip() for s in symptoms_str.split(',') if s.strip()]
        patient_info = {}
        request_metadata = {}
    else:
        meta_input = data.get('meta_input', {})
        symptom_list = data.get('symptoms', [])
        patient_info = data.get('patient_info', {})
        request_metadata = data.get('metadata', {})
        if meta_input:
            symptom_list = meta_input.get('symptoms', symptom_list)
            patient_info = meta_input.get('patient_info', patient_info)
            request_metadata = meta_input.get('metadata', request_metadata)
    
    if not symptom_list:
        return jsonify({"error": "No symptoms provided", "success": False}), 400
    
    # Filter valid symptoms
    valid_symptoms = [s for s in symptom_list if s in symptoms_dict]
    invalid_symptoms = [s for s in symptom_list if s not in symptoms_dict]
    
    if not valid_symptoms:
        return jsonify({
            "error": "No valid symptoms found",
            "invalid_symptoms": invalid_symptoms,
            "success": False
        }), 400
    
    # Predict disease
    predicted_disease, model_confidence = get_predicted_value(valid_symptoms)
    dis_des, pre, med, die, wrkout = helper(predicted_disease)
    severity_score = get_severity_score(valid_symptoms)
    
    # Format precautions
    precaution_list = []
    if len(pre) > 0:
        precaution_list = [str(p) for p in pre[0] if str(p) != 'nan']
    
    # Format medications
    med_list = []
    for m in med:
        if isinstance(m, str):
            try:
                parsed = eval(m) if m.startswith('[') else [m]
                med_list.extend(parsed)
            except:
                med_list.append(m)
    
    # Format diets
    diet_list = []
    for d in die:
        if isinstance(d, str):
            try:
                parsed = eval(d) if d.startswith('[') else [d]
                diet_list.extend(parsed)
            except:
                diet_list.append(d)
    
    # Format workouts
    workout_list = [str(w) for w in wrkout.values] if not wrkout.empty else []
    
    # Prefer model-derived confidence when available.
    confidence = model_confidence if model_confidence is not None else min(95.0, 60.0 + (len(valid_symptoms) * 8.5))
    
    result = {
        "success": True,
        "prediction": {
            "disease": predicted_disease,
            "confidence": round(confidence, 1),
            "severity_score": severity_score,
            "description": dis_des,
            "precautions": precaution_list,
            "medications": med_list,
            "diets": diet_list,
            "workouts": workout_list
        },
        "input": {
            "symptoms": valid_symptoms,
            "invalid_symptoms": invalid_symptoms,
            "patient_info": patient_info
        },
        "meta": {
            "model": "Support Vector Machine (Linear Kernel)",
            "model_version": "1.0.0",
            "symptoms_analyzed": len(valid_symptoms),
            "total_symptoms_available": len(symptoms_dict),
            "total_diseases": len(diseases_list),
            "timestamp": datetime.datetime.utcnow().isoformat()
        },
        "meta_input": {
            "symptoms": valid_symptoms,
            "patient_info": patient_info,
            "metadata": request_metadata
        },
        "meta_output": {
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence, 1),
                "severity_score": severity_score
            },
            "model": {
                "name": "Support Vector Machine",
                "kernel": "linear",
                "model_version": "1.0.0"
            },
            "timing": {
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        }
    }
    
    # Save prediction to database if user is authenticated
    token = None
    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
    if not token and 'auth_token' in session:
        token = session['auth_token']
    
    if token:
        user_data = verify_jwt_token(token)
        if user_data:
            prediction_doc = {
                "user_id": user_data['user_id'],
                "symptoms": valid_symptoms,
                "disease": predicted_disease,
                "confidence": confidence,
                "severity_score": severity_score,
                "patient_info": patient_info,
                "created_at": datetime.datetime.utcnow()
            }
            if MONGO_AVAILABLE:
                predictions_collection.insert_one(prediction_doc)
                users_collection.update_one(
                    {"email": user_data['email']},
                    {"$inc": {"predictions_count": 1}}
                )
            else:
                _predictions_store.append(prediction_doc)
    
    return jsonify(result), 200


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Legacy form-based prediction (renders template)"""
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms" or not symptoms:
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        user_symptoms = [symptom for symptom in user_symptoms if symptom and symptom in symptoms_dict]
        
        if not user_symptoms:
            message = "Please enter valid symptoms. No valid symptoms were found."
            return render_template('index.html', message=message)
        
        predicted_disease, _ = get_predicted_value(user_symptoms)
        dis_des, pre, med, die, wrkout = helper(predicted_disease)
        
        my_precautions = []
        for i in pre[0]:
            my_precautions.append(i)
        
        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=med, my_diet=die,
                               workout=wrkout)
    
    return render_template('index.html')


# ============================================================
# History API Endpoints
# ============================================================
@app.route('/api/predictions/history', methods=['GET'])
@token_required
def get_prediction_history(user_data):
    """Get prediction history for authenticated user"""
    if MONGO_AVAILABLE:
        history = list(predictions_collection.find(
            {"user_id": user_data['user_id']},
            {"_id": 0}
        ).sort("created_at", -1).limit(50))
        # Convert datetime objects
        for h in history:
            if 'created_at' in h:
                h['created_at'] = h['created_at'].isoformat()
    else:
        history = [p for p in _predictions_store if p.get('user_id') == user_data['user_id']]
        for h in history:
            if 'created_at' in h and hasattr(h['created_at'], 'isoformat'):
                h['created_at'] = h['created_at'].isoformat()
    
    return jsonify({"success": True, "history": history, "total": len(history)}), 200


# ============================================================
# Contact API Endpoint
# ============================================================
@app.route('/api/contact', methods=['POST'])
def submit_contact():
    """Submit contact form"""
    data = request.get_json() if request.is_json else request.form
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    message = data.get('message', '').strip()
    
    if not all([name, email, message]):
        return jsonify({"error": "All fields are required", "success": False}), 400
    
    contact_doc = {
        "name": name,
        "email": email,
        "message": message,
        "created_at": datetime.datetime.utcnow()
    }
    
    if MONGO_AVAILABLE:
        contacts_collection.insert_one(contact_doc)
    else:
        _contacts_store.append(contact_doc)
    
    return jsonify({"success": True, "message": "Message sent successfully"}), 201


# ============================================================
# Health Check & Stats API
# ============================================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        "status": "healthy",
        "database": "connected" if MONGO_AVAILABLE else "fallback_mode",
        "model_loaded": True,
        "symptoms_available": len(symptoms_dict),
        "diseases_available": len(diseases_list),
        "version": "2.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }), 200


@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get list of all diseases the model can predict"""
    disease_data = []
    for idx, name in sorted(diseases_list.items()):
        disease_data.append({"id": idx, "name": name.strip()})
    return jsonify({"success": True, "diseases": disease_data, "total": len(disease_data)}), 200


# ============================================================
# Run Application
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    # Werkzeug's auto-reloader can raise WinError 10038 on some Windows/Python setups.
    use_reloader = False if (os.name == "nt" and debug) else debug
    print(f"")
    print(f"== Health SymptomSense Server starting... ==")
    print(f"   URL: http://localhost:{port}")
    print(f"   Debug: {debug}")
    print(f"   Database: {'MongoDB' if MONGO_AVAILABLE else 'In-Memory Fallback'}")
    print(f"   ML Model: SVM Loaded ({len(symptoms_dict)} symptoms, {len(diseases_list)} diseases)")
    print(f"")
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
