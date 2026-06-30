"""Smart triage: classify urgency from symptoms and severity."""

CRITICAL_SYMPTOMS = {
    "chest_pain", "breathlessness", "fast_heart_rate", "coma", "stomach_bleeding",
    "bloody_stool", "acute_liver_failure", "weakness_of_one_body_side", "slurred_speech",
    "altered_sensorium",
}

HIGH_SEVERITY_SYMPTOMS = {
    "high_fever", "blood_in_sputum", "palpitations", "dizziness", "neck_pain",
    "stiff_neck", "severe headache", "headache",
}


def classify_triage(symptoms, severity_score):
    """
    Classify patient into triage levels:
    - URGENT: seek emergency care immediately
    - MODERATE: consult doctor within 24-48 hours
    - ROUTINE: monitor and self-care with precautions
    """
    symptom_set = {s.lower().replace(" ", "_") for s in symptoms}
    critical_hits = len(symptom_set & CRITICAL_SYMPTOMS)

    reasons = []
    if critical_hits >= 1:
        reasons.append(f"{critical_hits} critical symptom(s) detected")
        return _triage_result("URGENT", "Seek emergency medical care immediately", reasons, severity_score)

    high_hits = sum(1 for s in symptoms if s.replace("_", " ") in HIGH_SEVERITY_SYMPTOMS or s in HIGH_SEVERITY_SYMPTOMS)
    if severity_score >= 75 or high_hits >= 2:
        reasons.append("High severity score or multiple concerning symptoms")
        return _triage_result("MODERATE", "Consult a healthcare provider within 24-48 hours", reasons, severity_score)

    if severity_score >= 50 or high_hits >= 1:
        reasons.append("Moderate symptom severity detected")
        return _triage_result("MODERATE", "Schedule a doctor visit if symptoms persist or worsen", reasons, severity_score)

    reasons.append("Symptoms appear manageable with self-care")
    return _triage_result("ROUTINE", "Monitor symptoms and follow recommended precautions", reasons, severity_score)


def _triage_result(level, action, reasons, severity_score):
    color_map = {"URGENT": "#ef4444", "MODERATE": "#f59e0b", "ROUTINE": "#10b981"}
    icon_map = {"URGENT": "alert-triangle", "MODERATE": "clock", "ROUTINE": "check-circle"}
    return {
        "level": level,
        "action": action,
        "reasons": reasons,
        "severity_score": severity_score,
        "color": color_map.get(level, "#6366f1"),
        "icon": icon_map.get(level, "info"),
        "priority": {"URGENT": 1, "MODERATE": 2, "ROUTINE": 3}.get(level, 3),
    }
