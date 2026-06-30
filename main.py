"""
Health SymptomSense - Personalized Medical Recommendation System
Production-ready Flask backend with ML, MongoDB, Redis, JWT, and analytics.
"""

import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from functools import wraps
from urllib.parse import urlparse

from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file
from flask_cors import CORS
import io

from config import Config
from extensions import init_extensions
from services import analytics as analytics_service
from services import cache as cache_service
from services import ml_engine
from services import report as report_service
from services import triage as triage_service

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)
limiter = init_extensions(app)

# ============================================================
# MongoDB
# ============================================================
try:
    from pymongo import MongoClient

    uri_db_name = urlparse(Config.MONGODB_URI).path.lstrip("/")
    mongo_db_name = Config.MONGODB_DB_NAME or (uri_db_name if uri_db_name else "health_symptomsense")
    client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[mongo_db_name]
    users_collection = db["users"]
    predictions_collection = db["predictions"]
    contacts_collection = db["contacts"]
    users_collection.create_index("email", unique=True)
    predictions_collection.create_index([("user_id", 1), ("created_at", -1)])
    MONGO_AVAILABLE = True
    logger.info("MongoDB connected successfully")
except Exception as exc:
    MONGO_AVAILABLE = False
    logger.warning("MongoDB not available: %s — using in-memory fallback", exc)
    _users_store = {}
    _predictions_store = []
    _contacts_store = []

# ============================================================
# Redis Cache
# ============================================================
REDIS_AVAILABLE = cache_service.init_cache(Config.REDIS_URL)

# ============================================================
# ML Engine
# ============================================================
ml_engine.init_engine()
symptoms_dict = ml_engine.symptoms_dict
diseases_list = ml_engine.diseases_list

# ============================================================
# JWT Helpers
# ============================================================
JWT_SECRET = Config.JWT_SECRET_KEY
JWT_EXPIRY = Config.JWT_ACCESS_TOKEN_EXPIRES


def create_jwt_token(user_id, email, name):
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode()
    payload_data = {
        "user_id": str(user_id),
        "email": email,
        "name": name,
        "exp": int(time.time()) + JWT_EXPIRY,
        "iat": int(time.time()),
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode()
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{signature}"


def verify_jwt_token(token):
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        expected_sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload_data = json.loads(base64.urlsafe_b64decode(payload + "=="))
        if payload_data.get("exp", 0) < int(time.time()):
            return None
        return payload_data
    except Exception:
        return None


def hash_password(password):
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password, stored_hash):
    try:
        salt, hashed = stored_hash.split(":")
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(hashed, expected.hex())
    except Exception:
        return False


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        if not token and "auth_token" in session:
            token = session["auth_token"]
        if not token:
            return jsonify({"error": "Authentication required", "authenticated": False}), 401
        user_data = verify_jwt_token(token)
        if not user_data:
            return jsonify({"error": "Invalid or expired token", "authenticated": False}), 401
        return f(user_data, *args, **kwargs)

    return decorated


def _get_auth_token():
    if "Authorization" in request.headers:
        auth_header = request.headers["Authorization"]
        if auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
    return session.get("auth_token")


def _format_recommendations(disease):
    info = ml_engine.format_disease_recommendations(disease)
    return info["description"], info["precautions"], info["medications"], info["diets"], info["workouts"]


# ============================================================
# Security Headers
# ============================================================
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if not Config.FLASK_DEBUG:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ============================================================
# Page Routes
# ============================================================
@app.route("/")
def index():
    user = None
    if "auth_token" in session:
        user_data = verify_jwt_token(session["auth_token"])
        if user_data:
            user = user_data
    return render_template("index.html", user=user)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


# ============================================================
# Auth API
# ============================================================
@app.route("/api/auth/register", methods=["POST"])
@limiter.limit("10 per minute")
def register():
    data = request.get_json() if request.is_json else request.form
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([name, email, password]):
        return jsonify({"error": "All fields are required", "success": False}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters", "success": False}), 400

    hashed_pw = hash_password(password)

    if MONGO_AVAILABLE:
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered", "success": False}), 409
        user_doc = {
            "name": name,
            "email": email,
            "password": hashed_pw,
            "avatar": name[0].upper(),
            "created_at": datetime.datetime.utcnow(),
            "predictions_count": 0,
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
            "predictions_count": 0,
        }

    token = create_jwt_token(user_id, email, name)
    session["auth_token"] = token
    return jsonify({
        "success": True,
        "message": "Registration successful",
        "token": token,
        "user": {"id": user_id, "name": name, "email": email, "avatar": name[0].upper()},
    }), 201


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("15 per minute")
def login():
    data = request.get_json() if request.is_json else request.form
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([email, password]):
        return jsonify({"error": "Email and password are required", "success": False}), 400

    if MONGO_AVAILABLE:
        user = users_collection.find_one({"email": email})
        if not user or not verify_password(password, user["password"]):
            return jsonify({"error": "Invalid email or password", "success": False}), 401
        user_id = str(user["_id"])
        name = user["name"]
        avatar = user.get("avatar", name[0].upper())
        pred_count = user.get("predictions_count", 0)
    else:
        user = _users_store.get(email)
        if not user or not verify_password(password, user["password"]):
            return jsonify({"error": "Invalid email or password", "success": False}), 401
        user_id = user["id"]
        name = user["name"]
        avatar = user.get("avatar", name[0].upper())
        pred_count = user.get("predictions_count", 0)

    token = create_jwt_token(user_id, email, name)
    session["auth_token"] = token
    return jsonify({
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user_id, "name": name, "email": email,
            "avatar": avatar, "predictions_count": pred_count,
        },
    }), 200


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.pop("auth_token", None)
    return jsonify({"success": True, "message": "Logged out successfully"}), 200


@app.route("/api/auth/me", methods=["GET"])
@token_required
def get_current_user(user_data):
    return jsonify({
        "success": True,
        "user": {
            "id": user_data["user_id"],
            "name": user_data["name"],
            "email": user_data["email"],
            "avatar": user_data["name"][0].upper(),
        },
    }), 200


# ============================================================
# Prediction API
# ============================================================
@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    cache_key = "api:symptoms"
    cached = cache_service.cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    symptom_list = ml_engine.get_all_symptoms_metadata()
    response = {"success": True, "symptoms": symptom_list, "total": len(symptom_list)}
    cache_service.cache_set(cache_key, response, Config.CACHE_TTL)
    return jsonify(response), 200


@app.route("/api/symptoms/related", methods=["POST"])
@limiter.limit("30 per minute")
def get_related_symptoms():
    data = request.get_json() or {}
    selected = data.get("symptoms", [])
    if not selected:
        return jsonify({"error": "No symptoms provided", "success": False}), 400
    valid = [s for s in selected if s in symptoms_dict]
    related = ml_engine.get_related_symptoms(valid)
    return jsonify({"success": True, "related_symptoms": related, "total": len(related)}), 200


@app.route("/api/predict", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT_PREDICT)
def api_predict():
    data = request.get_json() if request.is_json else None

    if not data:
        symptoms_str = request.form.get("symptoms", "")
        symptom_list = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        patient_info = {}
        request_metadata = {}
    else:
        meta_input = data.get("meta_input", {})
        symptom_list = data.get("symptoms", [])
        patient_info = data.get("patient_info", {})
        request_metadata = data.get("metadata", {})
        if meta_input:
            symptom_list = meta_input.get("symptoms", symptom_list)
            patient_info = meta_input.get("patient_info", patient_info)
            request_metadata = meta_input.get("metadata", request_metadata)

    if not symptom_list:
        return jsonify({"error": "No symptoms provided", "success": False}), 400

    valid_symptoms = [s for s in symptom_list if s in symptoms_dict]
    invalid_symptoms = [s for s in symptom_list if s not in symptoms_dict]

    if not valid_symptoms:
        return jsonify({
            "error": "No valid symptoms found",
            "invalid_symptoms": invalid_symptoms,
            "success": False,
        }), 400

    predicted_disease, model_confidence = ml_engine.get_predicted_value(valid_symptoms)
    dis_des, pre, med, die, wrkout = _format_recommendations(predicted_disease)
    severity_score = ml_engine.get_severity_score(valid_symptoms)
    differential = ml_engine.get_top_k_predictions(valid_symptoms, k=3)
    triage = triage_service.classify_triage(valid_symptoms, severity_score)
    related_symptoms = ml_engine.get_related_symptoms(valid_symptoms, limit=5)

    confidence = model_confidence if model_confidence is not None else min(95.0, 60.0 + (len(valid_symptoms) * 8.5))

    result = {
        "success": True,
        "prediction": {
            "disease": predicted_disease,
            "confidence": round(confidence, 1),
            "severity_score": severity_score,
            "description": dis_des,
            "precautions": pre,
            "medications": med,
            "diets": die,
            "workouts": wrkout,
        },
        "differential_diagnosis": differential,
        "triage": triage,
        "related_symptoms": related_symptoms,
        "input": {
            "symptoms": valid_symptoms,
            "invalid_symptoms": invalid_symptoms,
            "patient_info": patient_info,
        },
        "meta": {
            "model": "Support Vector Machine (Linear Kernel)",
            "model_version": "1.0.0",
            "symptoms_analyzed": len(valid_symptoms),
            "total_symptoms_available": len(symptoms_dict),
            "total_diseases": len(diseases_list),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "app_version": Config.APP_VERSION,
        },
        "meta_input": {
            "symptoms": valid_symptoms,
            "patient_info": patient_info,
            "metadata": request_metadata,
        },
        "meta_output": {
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence, 1),
                "severity_score": severity_score,
            },
            "triage_level": triage["level"],
            "model": {"name": "Support Vector Machine", "kernel": "linear", "model_version": "1.0.0"},
            "timing": {"timestamp": datetime.datetime.utcnow().isoformat()},
        },
    }

    token = _get_auth_token()
    if token:
        user_data = verify_jwt_token(token)
        if user_data:
            prediction_doc = {
                "user_id": user_data["user_id"],
                "symptoms": valid_symptoms,
                "disease": predicted_disease,
                "confidence": confidence,
                "severity_score": severity_score,
                "triage_level": triage["level"],
                "patient_info": patient_info,
                "differential_diagnosis": differential,
                "created_at": datetime.datetime.utcnow(),
            }
            if MONGO_AVAILABLE:
                predictions_collection.insert_one(prediction_doc)
                users_collection.update_one(
                    {"email": user_data["email"]},
                    {"$inc": {"predictions_count": 1}},
                )
            else:
                _predictions_store.append(prediction_doc)

    return jsonify(result), 200


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        if symptoms == "Symptoms" or not symptoms:
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template("index.html", message=message)

        user_symptoms = [s.strip() for s in symptoms.split(",")]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        user_symptoms = [symptom for symptom in user_symptoms if symptom and symptom in symptoms_dict]

        if not user_symptoms:
            message = "Please enter valid symptoms. No valid symptoms were found."
            return render_template("index.html", message=message)

        predicted_disease, _ = ml_engine.get_predicted_value(user_symptoms)
        info = ml_engine.format_disease_recommendations(predicted_disease)
        my_precautions = info["precautions"]

        return render_template(
            "index.html",
            predicted_disease=predicted_disease,
            dis_des=info["description"],
            my_precautions=my_precautions,
            medications=info["medications"],
            my_diet=info["diets"],
            workout=info["workouts"],
        )

    return render_template("index.html")


# ============================================================
# History & Analytics API
# ============================================================
@app.route("/api/predictions/history", methods=["GET"])
@token_required
def get_prediction_history(user_data):
    if MONGO_AVAILABLE:
        history = list(predictions_collection.find(
            {"user_id": user_data["user_id"]},
            {"_id": 0},
        ).sort("created_at", -1).limit(50))
        for h in history:
            if "created_at" in h:
                h["created_at"] = h["created_at"].isoformat()
    else:
        history = [p for p in _predictions_store if p.get("user_id") == user_data["user_id"]]
        for h in history:
            if "created_at" in h and hasattr(h["created_at"], "isoformat"):
                h["created_at"] = h["created_at"].isoformat()

    return jsonify({"success": True, "history": history, "total": len(history)}), 200


@app.route("/api/predictions/analytics", methods=["GET"])
@token_required
def get_prediction_analytics(user_data):
    if MONGO_AVAILABLE:
        history = list(predictions_collection.find(
            {"user_id": user_data["user_id"]},
            {"_id": 0},
        ).sort("created_at", -1).limit(100))
        for h in history:
            if "created_at" in h:
                h["created_at"] = h["created_at"].isoformat()
    else:
        history = [p for p in _predictions_store if p.get("user_id") == user_data["user_id"]]
        for h in history:
            if "created_at" in h and hasattr(h["created_at"], "isoformat"):
                h["created_at"] = h["created_at"].isoformat()

    analytics = analytics_service.compute_analytics(history)
    return jsonify({"success": True, "analytics": analytics}), 200


@app.route("/api/report/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_report():
    data = request.get_json() or {}
    if not data.get("prediction"):
        return jsonify({"error": "Prediction data required", "success": False}), 400

    try:
        pdf_bytes = report_service.generate_health_report(
            data,
            patient_info=data.get("input", {}).get("patient_info"),
        )
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"health_report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf",
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        return jsonify({"error": f"Report generation failed: {exc}", "success": False}), 500


# ============================================================
# Contact API
# ============================================================
@app.route("/api/contact", methods=["POST"])
@limiter.limit("5 per minute")
def submit_contact():
    data = request.get_json() if request.is_json else request.form
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    message = data.get("message", "").strip()

    if not all([name, email, message]):
        return jsonify({"error": "All fields are required", "success": False}), 400

    contact_doc = {
        "name": name,
        "email": email,
        "message": message,
        "created_at": datetime.datetime.utcnow(),
    }

    if MONGO_AVAILABLE:
        contacts_collection.insert_one(contact_doc)
    else:
        _contacts_store.append(contact_doc)

    return jsonify({"success": True, "message": "Message sent successfully"}), 201


# ============================================================
# Health & Stats API
# ============================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "database": "connected" if MONGO_AVAILABLE else "fallback_mode",
        "cache": "redis" if REDIS_AVAILABLE else "memory",
        "model_loaded": True,
        "symptoms_available": len(symptoms_dict),
        "diseases_available": len(diseases_list),
        "version": Config.APP_VERSION,
        "features": [
            "differential_diagnosis",
            "smart_triage",
            "health_analytics",
            "related_symptoms",
            "pdf_reports",
            "rate_limiting",
            "prometheus_metrics",
        ],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }), 200


@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    cache_key = "api:diseases"
    cached = cache_service.cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    disease_data = []
    for idx, name in sorted(diseases_list.items()):
        disease_data.append({"id": idx, "name": name.strip()})
    response = {"success": True, "diseases": disease_data, "total": len(disease_data)}
    cache_service.cache_set(cache_key, response, Config.CACHE_TTL)
    return jsonify(response), 200


# ============================================================
# Run Application
# ============================================================
if __name__ == "__main__":
    use_reloader = False if (os.name == "nt" and Config.FLASK_DEBUG) else Config.FLASK_DEBUG
    print("")
    print("== Health SymptomSense Server v{} ==".format(Config.APP_VERSION))
    print(f"   URL: http://localhost:{Config.PORT}")
    print(f"   Debug: {Config.FLASK_DEBUG}")
    print(f"   Database: {'MongoDB' if MONGO_AVAILABLE else 'In-Memory Fallback'}")
    print(f"   Cache: {'Redis' if REDIS_AVAILABLE else 'In-Memory'}")
    print(f"   ML Model: SVM ({len(symptoms_dict)} symptoms, {len(diseases_list)} diseases)")
    print(f"   Metrics: http://localhost:{Config.PORT}/metrics")
    print("")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.FLASK_DEBUG, use_reloader=use_reloader)
