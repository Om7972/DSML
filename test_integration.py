"""Full integration test for Health SymptomSense with MongoDB"""
import urllib.request
import json

BASE = "http://127.0.0.1:5000"

def api(method, path, data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req)
        return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

print("=" * 55)
print("  HEALTH SYMPTOMSENSE - FULL INTEGRATION TEST")
print("=" * 55)

# 1. Health Check
data, code = api("GET", "/api/health")
print("\n[1] HEALTH CHECK", code)
print("    Status:", data["status"])
print("    DB:", data["database"])
print("    Model:", data["model_loaded"])
print("    Symptoms:", data["symptoms_available"], "| Diseases:", data["diseases_available"])

# 2. Register new test user
email = "integration_test_user@test.com"
data, code = api("POST", "/api/auth/register", {
    "name": "Integration Test User",
    "email": email,
    "password": "testpass123"
})
print("\n[2] REGISTER", code)
if code == 201:
    print("    User created:", data["user"]["name"], "-", data["user"]["email"])
    token = data["token"]
elif code == 409:
    print("    User already exists, logging in instead...")
    data, code = api("POST", "/api/auth/login", {"email": email, "password": "testpass123"})
    print("    Login:", code, data["user"]["name"])
    token = data["token"]
else:
    print("    Error:", data)
    token = None

# 3. Get user profile
if token:
    data, code = api("GET", "/api/auth/me", token=token)
    print("\n[3] USER PROFILE", code)
    print("    Name:", data["user"]["name"])
    print("    Email:", data["user"]["email"])
    print("    Avatar:", data["user"]["avatar"])

# 4. Predict disease (authenticated - stores in history)
data, code = api("POST", "/api/predict", {
    "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
    "patient_info": {"age": 25, "gender": "male"}
}, token=token)
print("\n[4] PREDICTION (Authenticated)", code)
print("    Disease:", data["prediction"]["disease"])
print("    Confidence:", data["prediction"]["confidence"], "%")
print("    Severity:", data["prediction"]["severity_score"])
print("    Medications:", data["prediction"]["medications"][:3])
print("    Precautions:", data["prediction"]["precautions"][:2])
print("    Diets:", data["prediction"]["diets"][:2])
print("    Workouts:", data["prediction"]["workouts"][:2])

# 5. Predict another disease
data2, code2 = api("POST", "/api/predict", {
    "symptoms": ["stomach_pain", "acidity", "vomiting", "indigestion"],
}, token=token)
print("\n[5] PREDICTION 2", code2)
print("    Disease:", data2["prediction"]["disease"])
print("    Confidence:", data2["prediction"]["confidence"], "%")

# 6. Check prediction history (should have entries now)
if token:
    data, code = api("GET", "/api/predictions/history", token=token)
    print("\n[6] PREDICTION HISTORY", code)
    print("    Total predictions stored:", data["total"])
    if data.get("history"):
        latest = data["history"][0]
        print("    Latest:", latest["disease"], "from", latest["symptoms"][:3])

# 7. Symptoms API
data, code = api("GET", "/api/symptoms")
print("\n[7] SYMPTOMS API", code)
print("    Total:", data["total"])
print("    Sample:", data["symptoms"][0]["name"], "(severity:", str(data["symptoms"][0]["severity"]) + ")")

# 8. Diseases API
data, code = api("GET", "/api/diseases")
print("\n[8] DISEASES API", code)
print("    Total:", data["total"])
print("    Sample:", data["diseases"][:5])

# 9. Contact form
data, code = api("POST", "/api/contact", {
    "name": "Test Contact",
    "email": "contact@test.com",
    "message": "Great app! Integration test message."
})
print("\n[9] CONTACT FORM", code)
print("    Submitted:", data["success"])

# 10. Page routes
for page in ["/", "/about", "/contact", "/developer", "/blog"]:
    req = urllib.request.Request(BASE + page)
    r = urllib.request.urlopen(req)
    print(f"\n[10] PAGE {page} -> {r.status} ({len(r.read())} bytes)")

print("\n" + "=" * 55)
print("  ALL TESTS COMPLETED SUCCESSFULLY!")
print("  Data is now visible in MongoDB Compass")
print("  DB: health_symptomsense")
print("  Collections: users, predictions, contacts")
print("=" * 55)
