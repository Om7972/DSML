"""Machine learning engine: prediction, differential diagnosis, related symptoms."""

import os
import pickle

import numpy as np
import pandas as pd


symptoms_dict = {
    "itching": 0, "skin_rash": 1, "nodal_skin_eruptions": 2, "continuous_sneezing": 3,
    "shivering": 4, "chills": 5, "joint_pain": 6, "stomach_pain": 7, "acidity": 8,
    "ulcers_on_tongue": 9, "muscle_wasting": 10, "vomiting": 11, "burning_micturition": 12,
    "spotting_ urination": 13, "fatigue": 14, "weight_gain": 15, "anxiety": 16,
    "cold_hands_and_feets": 17, "mood_swings": 18, "weight_loss": 19, "restlessness": 20,
    "lethargy": 21, "patches_in_throat": 22, "irregular_sugar_level": 23, "cough": 24,
    "high_fever": 25, "sunken_eyes": 26, "breathlessness": 27, "sweating": 28,
    "dehydration": 29, "indigestion": 30, "headache": 31, "yellowish_skin": 32,
    "dark_urine": 33, "nausea": 34, "loss_of_appetite": 35, "pain_behind_the_eyes": 36,
    "back_pain": 37, "constipation": 38, "abdominal_pain": 39, "diarrhoea": 40,
    "mild_fever": 41, "yellow_urine": 42, "yellowing_of_eyes": 43, "acute_liver_failure": 44,
    "fluid_overload": 45, "swelling_of_stomach": 46, "swelled_lymph_nodes": 47, "malaise": 48,
    "blurred_and_distorted_vision": 49, "phlegm": 50, "throat_irritation": 51,
    "redness_of_eyes": 52, "sinus_pressure": 53, "runny_nose": 54, "congestion": 55,
    "chest_pain": 56, "weakness_in_limbs": 57, "fast_heart_rate": 58,
    "pain_during_bowel_movements": 59, "pain_in_anal_region": 60, "bloody_stool": 61,
    "irritation_in_anus": 62, "neck_pain": 63, "dizziness": 64, "cramps": 65, "bruising": 66,
    "obesity": 67, "swollen_legs": 68, "swollen_blood_vessels": 69, "puffy_face_and_eyes": 70,
    "enlarged_thyroid": 71, "brittle_nails": 72, "swollen_extremeties": 73, "excessive_hunger": 74,
    "extra_marital_contacts": 75, "drying_and_tingling_lips": 76, "slurred_speech": 77,
    "knee_pain": 78, "hip_joint_pain": 79, "muscle_weakness": 80, "stiff_neck": 81,
    "swelling_joints": 82, "movement_stiffness": 83, "spinning_movements": 84,
    "loss_of_balance": 85, "unsteadiness": 86, "weakness_of_one_body_side": 87,
    "loss_of_smell": 88, "bladder_discomfort": 89, "foul_smell_of urine": 90,
    "continuous_feel_of_urine": 91, "passage_of_gases": 92, "internal_itching": 93,
    "toxic_look_(typhos)": 94, "depression": 95, "irritability": 96, "muscle_pain": 97,
    "altered_sensorium": 98, "red_spots_over_body": 99, "belly_pain": 100,
    "abnormal_menstruation": 101, "dischromic _patches": 102, "watering_from_eyes": 103,
    "increased_appetite": 104, "polyuria": 105, "family_history": 106, "mucoid_sputum": 107,
    "rusty_sputum": 108, "lack_of_concentration": 109, "visual_disturbances": 110,
    "receiving_blood_transfusion": 111, "receiving_unsterile_injections": 112, "coma": 113,
    "stomach_bleeding": 114, "distention_of_abdomen": 115, "history_of_alcohol_consumption": 116,
    "fluid_overload.1": 117, "blood_in_sputum": 118, "prominent_veins_on_calf": 119,
    "palpitations": 120, "painful_walking": 121, "pus_filled_pimples": 122, "blackheads": 123,
    "scurring": 124, "skin_peeling": 125, "silver_like_dusting": 126, "small_dents_in_nails": 127,
    "inflammatory_nails": 128, "blister": 129, "red_sore_around_nose": 130, "yellow_crust_ooze": 131,
}

diseases_list = {
    15: "Fungal infection", 4: "Allergy", 16: "GERD", 9: "Chronic cholestasis",
    14: "Drug Reaction", 33: "Peptic ulcer diseae", 1: "AIDS", 12: "Diabetes ",
    17: "Gastroenteritis", 6: "Bronchial Asthma", 23: "Hypertension ", 30: "Migraine",
    7: "Cervical spondylosis", 32: "Paralysis (brain hemorrhage)", 28: "Jaundice",
    29: "Malaria", 8: "Chicken pox", 11: "Dengue", 37: "Typhoid", 40: "hepatitis A",
    19: "Hepatitis B", 20: "Hepatitis C", 21: "Hepatitis D", 22: "Hepatitis E",
    3: "Alcoholic hepatitis", 36: "Tuberculosis", 10: "Common Cold", 34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)", 18: "Heart attack", 39: "Varicose veins",
    26: "Hypothyroidism", 24: "Hyperthyroidism", 25: "Hypoglycemia", 31: "Osteoarthristis",
    5: "Arthritis", 0: "(vertigo) Paroymsal  Positional Vertigo", 2: "Acne",
    38: "Urinary tract infection", 35: "Psoriasis", 27: "Impetigo",
}

_symptom_columns = None
_training_df = None
_svc = None
_severity_df = None
_description_df = None
_precautions_df = None
_medications_df = None
_diets_df = None
_workout_df = None


def _load_model():
    global _svc
    if _svc is not None:
        return _svc
    for model_path in ("models/svc.pkl", "models/svc .pkl"):
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                _svc = pickle.load(model_file)
            return _svc
    raise FileNotFoundError("No SVM model found. Run `python retrain.py` to generate models/svc.pkl.")


def _load_datasets():
    global _training_df, _symptom_columns, _severity_df
    global _description_df, _precautions_df, _medications_df, _diets_df, _workout_df

    if _training_df is None:
        _training_df = pd.read_csv("datasets/Training.csv")
        _symptom_columns = [c for c in _training_df.columns if c != "prognosis"]
        _severity_df = pd.read_csv("datasets/Symptom-severity.csv")
        _description_df = pd.read_csv("datasets/description.csv")
        _precautions_df = pd.read_csv("datasets/precautions_df.csv")
        _medications_df = pd.read_csv("datasets/medications.csv")
        _diets_df = pd.read_csv("datasets/diets.csv")
        _workout_df = pd.read_csv("datasets/workout_df.csv")


def init_engine():
    """Load model and datasets at startup."""
    _load_model()
    _load_datasets()


def build_input_vector(symptom_list):
    vector = np.zeros(len(symptoms_dict))
    for item in symptom_list:
        if item in symptoms_dict:
            vector[symptoms_dict[item]] = 1
    return vector


def get_severity_score(symptom_list):
    _load_datasets()
    total = 0
    count = 0
    for s in symptom_list:
        match = _severity_df[_severity_df["Symptom"].str.strip().str.lower() == s.strip().lower()]
        if not match.empty:
            total += match["weight"].values[0]
            count += 1
    return round((total / max(count, 1)) * 15, 1)


def get_disease_info(disease):
    _load_datasets()
    desc = _description_df[_description_df["Disease"] == disease]["Description"]
    desc = " ".join([w for w in desc])

    pre = _precautions_df[_precautions_df["Disease"] == disease][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ]
    pre = [col for col in pre.values]

    med = _medications_df[_medications_df["Disease"] == disease]["Medication"]
    med = [m for m in med.values]

    die = _diets_df[_diets_df["Disease"] == disease]["Diet"]
    die = [d for d in die.values]

    wrkout = _workout_df[_workout_df["disease"] == disease]["workout"]
    return desc, pre, med, die, wrkout


def _format_list_items(items):
    result = []
    for item in items:
        if isinstance(item, str):
            try:
                parsed = eval(item) if item.startswith("[") else [item]
                result.extend(parsed)
            except Exception:
                result.append(item)
        else:
            result.append(str(item))
    return result


def format_disease_recommendations(disease):
    desc, pre, med, die, wrkout = get_disease_info(disease)
    precaution_list = []
    if len(pre) > 0:
        precaution_list = [str(p) for p in pre[0] if str(p) != "nan"]
    workout_list = [str(w) for w in wrkout.values] if not wrkout.empty else []
    return {
        "description": desc,
        "precautions": precaution_list,
        "medications": _format_list_items(med),
        "diets": _format_list_items(die),
        "workouts": workout_list,
    }


def _scores_to_confidence(scores):
    """Convert decision scores to relative confidence among candidates (sums to 100%)."""
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    return probs * 100.0


def get_top_k_predictions(symptom_list, k=3):
    """Return top-k differential diagnosis with confidence scores."""
    svc = _load_model()
    input_vector = build_input_vector(symptom_list)

    if hasattr(svc, "predict_proba"):
        proba = svc.predict_proba([input_vector])[0]
        top_indices = np.argsort(proba)[-k:][::-1]
        confidences = proba[top_indices] * 100.0
    elif hasattr(svc, "decision_function"):
        scores = svc.decision_function([input_vector])
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        scores = scores[0]
        top_indices = np.argsort(scores)[-k:][::-1]
        confidences = _scores_to_confidence(scores[top_indices])
    else:
        predicted_idx = int(svc.predict([input_vector])[0])
        return [{
            "disease": diseases_list[predicted_idx],
            "confidence": 85.0,
            "rank": 1,
        }]

    results = []
    for rank, (idx, conf) in enumerate(zip(top_indices, confidences), start=1):
        disease_idx = int(idx)
        if disease_idx in diseases_list:
            results.append({
                "disease": diseases_list[disease_idx],
                "confidence": round(float(conf), 1),
                "rank": rank,
            })
    return results


def get_predicted_value(patient_symptoms):
    """Predict primary disease from symptoms."""
    top_k = get_top_k_predictions(patient_symptoms, k=3)
    if not top_k:
        svc = _load_model()
        input_vector = build_input_vector(patient_symptoms)
        predicted_idx = int(svc.predict([input_vector])[0])
        return diseases_list[predicted_idx], 85.0
    primary = top_k[0]
    return primary["disease"], primary["confidence"]


def get_symptom_severity(symptom_key):
    """Return severity weight (1-7) for a symptom key."""
    _load_datasets()
    display = symptom_key.replace("_", " ").strip().lower()
    match = _severity_df[_severity_df["Symptom"].str.strip().str.lower() == display]
    return int(match["weight"].values[0]) if not match.empty else 3


def get_all_symptoms_metadata():
    """Return full symptom list with severity for API."""
    _load_datasets()
    symptom_list = []
    for symptom, idx in sorted(symptoms_dict.items(), key=lambda x: x[1]):
        symptom_list.append({
            "id": idx,
            "key": symptom,
            "name": symptom.replace("_", " ").title(),
            "severity": get_symptom_severity(symptom),
        })
    return symptom_list


def get_related_symptoms(selected_symptoms, limit=6):
    """Suggest co-occurring symptoms from training data."""
    _load_datasets()
    if not selected_symptoms:
        return []

    selected_set = set(selected_symptoms)
    mask = pd.Series(False, index=_training_df.index)
    for symptom in selected_symptoms:
        if symptom in _training_df.columns:
            mask |= _training_df[symptom] == 1

    matching_rows = _training_df[mask]
    if matching_rows.empty:
        return []

    co_occurrence = {}
    for col in _symptom_columns:
        if col in selected_set:
            continue
        count = (matching_rows[col] == 1).sum()
        if count > 0:
            co_occurrence[col] = int(count)

    sorted_symptoms = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:limit]
    results = []
    for key, frequency in sorted_symptoms:
        display_name = key.replace("_", " ").title()
        match = _severity_df[_severity_df["Symptom"].str.strip().str.lower() == key.replace("_", " ").strip().lower()]
        weight = int(match["weight"].values[0]) if not match.empty else 3
        results.append({
            "key": key,
            "name": display_name,
            "severity": weight,
            "co_occurrence_rate": round(frequency / len(matching_rows) * 100, 1),
        })
    return results
