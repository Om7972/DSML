"""Health analytics from prediction history."""

from collections import Counter
from datetime import datetime, timedelta


def compute_analytics(history):
    """Build health dashboard analytics from user prediction history."""
    if not history:
        return {
            "total_predictions": 0,
            "avg_confidence": 0,
            "avg_severity": 0,
            "most_common_diseases": [],
            "severity_trend": [],
            "symptom_frequency": [],
            "triage_breakdown": {"URGENT": 0, "MODERATE": 0, "ROUTINE": 0},
            "recent_predictions": [],
        }

    confidences = [h.get("confidence", 0) for h in history]
    severities = [h.get("severity_score", 0) for h in history]
    diseases = [h.get("disease", "Unknown") for h in history]

    disease_counts = Counter(diseases)
    most_common = [
        {"disease": d, "count": c}
        for d, c in disease_counts.most_common(5)
    ]

    all_symptoms = []
    for h in history:
        all_symptoms.extend(h.get("symptoms", []))
    symptom_counts = Counter(all_symptoms)
    symptom_frequency = [
        {"symptom": s.replace("_", " ").title(), "key": s, "count": c}
        for s, c in symptom_counts.most_common(10)
    ]

    severity_trend = _build_severity_trend(history)

    triage_breakdown = {"URGENT": 0, "MODERATE": 0, "ROUTINE": 0}
    for h in history:
        level = h.get("triage_level", "ROUTINE")
        if level in triage_breakdown:
            triage_breakdown[level] += 1

    recent = []
    for h in history[:5]:
        created = h.get("created_at", "")
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        recent.append({
            "disease": h.get("disease"),
            "confidence": h.get("confidence"),
            "severity_score": h.get("severity_score"),
            "symptoms_count": len(h.get("symptoms", [])),
            "triage_level": h.get("triage_level", "ROUTINE"),
            "created_at": created,
        })

    return {
        "total_predictions": len(history),
        "avg_confidence": round(sum(confidences) / len(confidences), 1),
        "avg_severity": round(sum(severities) / len(severities), 1),
        "most_common_diseases": most_common,
        "severity_trend": severity_trend,
        "symptom_frequency": symptom_frequency,
        "triage_breakdown": triage_breakdown,
        "recent_predictions": recent,
    }


def _build_severity_trend(history, days=14):
    """Group severity scores by day for the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    daily = {}

    for h in history:
        created = h.get("created_at")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
            except ValueError:
                continue
        if not isinstance(created, datetime):
            continue
        if created < cutoff:
            continue
        day_key = created.strftime("%Y-%m-%d")
        if day_key not in daily:
            daily[day_key] = []
        daily[day_key].append(h.get("severity_score", 0))

    trend = []
    for day in sorted(daily.keys()):
        scores = daily[day]
        trend.append({
            "date": day,
            "avg_severity": round(sum(scores) / len(scores), 1),
            "predictions": len(scores),
        })
    return trend
