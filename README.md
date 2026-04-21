<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/MongoDB-7.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🏥 Health SymptomSense</h1>
<h3 align="center">Personalized Medical Recommendation System with Machine Learning</h3>
<p align="center"><em>AI-Powered Disease Prediction | Real-Time Health Insights | Secure Authentication</em></p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [SRS Document (Software Requirements Specification)](#srs-document-software-requirements-specification)
  - [1. Introduction](#1-introduction)
  - [2. Overall Description](#2-overall-description)
  - [3. System Features & Functional Requirements](#3-system-features--functional-requirements)
  - [4. External Interface Requirements](#4-external-interface-requirements)
  - [5. Non-Functional Requirements](#5-non-functional-requirements)
  - [6. System Architecture](#6-system-architecture)
- [Synopsis Document](#synopsis-document)
  - [Project Title & Abstract](#project-title--abstract)
  - [Problem Statement](#problem-statement)
  - [Objectives](#objectives)
  - [Scope](#scope)
  - [Methodology](#methodology)
  - [Technologies Used](#technologies-used)
  - [Expected Outcomes](#expected-outcomes)
  - [Project Timeline](#project-timeline)
  - [Conclusion](#conclusion)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Machine Learning Model](#machine-learning-model)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**Health SymptomSense** is a full-stack, AI-powered **Personalized Medical Recommendation System** that leverages **Machine Learning** (Support Vector Machine) to predict diseases based on user-reported symptoms. The system provides comprehensive health recommendations including disease descriptions, precautions, medications, dietary suggestions, and workout plans.

### Key Highlights

| Feature | Details |
|---------|---------|
| **ML Model** | Support Vector Machine (Linear Kernel) trained on 4,920 cases |
| **Diseases** | Predicts 41 different diseases |
| **Symptoms** | Analyzes 132 medically-recognized symptoms |
| **Accuracy** | ~95% prediction accuracy on test data |
| **Backend** | Flask REST API with JWT Authentication |
| **Database** | MongoDB for user data & prediction history |
| **Frontend** | Modern glassmorphism UI with micro-interactions |
| **Auth** | JWT token-based authentication with encrypted passwords |
| **Voice** | Web Speech API for voice-based symptom input |

---

# SRS Document (Software Requirements Specification)

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document provides a comprehensive description of the **Personalized Medical Recommendation System with Machine Learning** — "Health SymptomSense". It details the functional and non-functional requirements, system architecture, external interfaces, and design constraints for the complete web-based health analysis platform.

### 1.2 Scope

The Health SymptomSense system is a web application that:
- Accepts user symptoms through a searchable multi-select interface or voice input
- Predicts potential diseases using a trained SVM machine learning model
- Provides personalized health recommendations (medications, diets, workouts, precautions)
- Manages user accounts with secure JWT-based authentication
- Stores prediction history in a MongoDB database
- Exposes RESTful API endpoints with structured meta input/output format

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| **SVM** | Support Vector Machine — a supervised ML classification algorithm |
| **JWT** | JSON Web Token — standard for secure token-based authentication |
| **REST API** | Representational State Transfer Application Programming Interface |
| **CRUD** | Create, Read, Update, Delete operations |
| **SPA** | Single Page Application |
| **PBKDF2** | Password-Based Key Derivation Function 2 |
| **CORS** | Cross-Origin Resource Sharing |
| **ML** | Machine Learning |
| **NLP** | Natural Language Processing |
| **UI/UX** | User Interface / User Experience |

### 1.4 References

- IEEE 830-1998 — IEEE Recommended Practice for SRS
- WHO ICD-11 International Classification of Diseases
- scikit-learn Documentation (v1.3)
- Flask Documentation (v2.3)
- MongoDB Documentation (v7.0)

### 1.5 Document Overview

This SRS is organized as follows:
- **Section 2** — Overall system description, user classes, constraints
- **Section 3** — Detailed functional requirements for each module
- **Section 4** — External interface requirements (UI, API, Hardware, Software)
- **Section 5** — Non-functional requirements (performance, security, reliability)
- **Section 6** — System architecture and data flow diagrams

---

## 2. Overall Description

### 2.1 Product Perspective

Health SymptomSense is a standalone web application that integrates:
- **Frontend Layer** — Modern HTML/CSS/JavaScript with glassmorphism design
- **Backend Layer** — Flask (Python) REST API server
- **ML Engine** — Trained SVM model for disease classification
- **Database Layer** — MongoDB for persistent data storage
- **Authentication Layer** — JWT-based secure authentication system

### 2.2 Product Functions

The system provides the following major functions:

1. **User Registration & Authentication** — Secure signup/login with password hashing
2. **Symptom Input** — Searchable dropdown with 132 symptoms + voice recognition
3. **Disease Prediction** — ML-based classification returning predicted disease
4. **Health Recommendations** — Detailed output including:
   - Disease description
   - Precautionary measures (4 per disease)
   - Medication suggestions
   - Dietary recommendations
   - Workout/exercise plans
5. **Prediction History** — Tracked per user in MongoDB
6. **API Endpoints** — RESTful JSON API for external integration
7. **Contact System** — User feedback/contact form with database storage

### 2.3 User Classes and Characteristics

| User Class | Description | Technical Expertise |
|------------|-------------|-------------------|
| **General User** | Individuals seeking preliminary health insights | Low |
| **Authenticated User** | Registered users with prediction history tracking | Low-Medium |
| **Healthcare Professional** | Medical practitioners using as supplementary tool | Medium |
| **API Consumer/Developer** | External systems integrating via REST API | High |
| **System Administrator** | Manages deployment, database, and model updates | High |

### 2.4 Operating Environment

- **Server OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python Runtime**: Python 3.10+
- **Database**: MongoDB 6.0+ (local or Atlas cloud)
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 15+, Edge 90+
- **Network**: HTTP/HTTPS, localhost or cloud deployment

### 2.5 Design and Implementation Constraints

1. ML model is pre-trained (offline training) and loaded at server startup
2. Disease prediction is limited to 41 diseases in the training dataset
3. System is for informational purposes only — not a substitute for medical diagnosis
4. MongoDB must be running for full functionality (fallback: in-memory storage)
5. Voice recognition requires browser support for Web Speech API

### 2.6 Assumptions and Dependencies

- Users have a modern web browser with JavaScript enabled
- MongoDB server is accessible at the configured URI
- The SVM model file (`models/svc .pkl`) is present and valid
- All CSV dataset files are present in the `datasets/` directory
- Python dependencies are installed per `requirements.txt`

---

## 3. System Features & Functional Requirements

### 3.1 User Authentication System

**Priority**: High

| Req ID | Requirement | Description |
|--------|-------------|-------------|
| FR-AUTH-01 | User Registration | System shall allow users to register with name, email, and password |
| FR-AUTH-02 | Password Security | Passwords shall be hashed using PBKDF2-SHA256 with random salt |
| FR-AUTH-03 | User Login | System shall authenticate users with email and password |
| FR-AUTH-04 | JWT Token Generation | System shall issue JWT tokens upon successful authentication |
| FR-AUTH-05 | Token Validation | System shall validate JWT tokens for protected endpoints |
| FR-AUTH-06 | Session Management | System shall maintain session state using tokens and server sessions |
| FR-AUTH-07 | User Logout | System shall invalidate user session on logout |
| FR-AUTH-08 | Duplicate Prevention | System shall prevent duplicate email registrations |

### 3.2 Symptom Input Module

**Priority**: High

| Req ID | Requirement | Description |
|--------|-------------|-------------|
| FR-SYM-01 | Symptom Search | System shall provide real-time search across 132 symptoms |
| FR-SYM-02 | Symptom Selection | Users shall select symptoms via click from dropdown |
| FR-SYM-03 | Chip Display | Selected symptoms shall appear as removable chips/tags |
| FR-SYM-04 | Voice Input | System shall support voice-based symptom input via Web Speech API |
| FR-SYM-05 | Voice Matching | System shall auto-match spoken words to symptom database |
| FR-SYM-06 | Severity Display | Each symptom shall display its severity weight (1-7 scale) |
| FR-SYM-07 | Validation | System shall validate at least one symptom before prediction |

### 3.3 Disease Prediction Engine

**Priority**: Critical

| Req ID | Requirement | Description |
|--------|-------------|-------------|
| FR-PRED-01 | ML Prediction | System shall predict disease using SVM model from selected symptoms |
| FR-PRED-02 | Feature Vector | System shall convert symptoms to a 132-dimensional binary vector |
| FR-PRED-03 | Confidence Score | System shall calculate and display prediction confidence |
| FR-PRED-04 | Severity Score | System shall compute severity score from symptom weights |
| FR-PRED-05 | Disease Info | System shall return disease description from dataset |
| FR-PRED-06 | Precautions | System shall return up to 4 precautions per predicted disease |
| FR-PRED-07 | Medications | System shall return recommended medications |
| FR-PRED-08 | Diet Plan | System shall return dietary recommendations |
| FR-PRED-09 | Workout Plan | System shall return recommended exercises/workouts |
| FR-PRED-10 | History Storage | Predictions shall be stored in database for authenticated users |

### 3.4 API System (Meta Input/Output)

**Priority**: High

| Req ID | Requirement | Description |
|--------|-------------|-------------|
| FR-API-01 | REST Endpoints | System shall expose RESTful JSON API endpoints |
| FR-API-02 | Meta Input | API shall accept structured JSON input with symptoms and patient_info |
| FR-API-03 | Meta Output | API shall return structured JSON with prediction, input, and meta fields |
| FR-API-04 | CORS Support | API shall support Cross-Origin requests |
| FR-API-05 | Error Handling | API shall return appropriate HTTP status codes and error messages |
| FR-API-06 | Health Check | System shall expose a `/api/health` endpoint for monitoring |

### 3.5 User Interface

**Priority**: High

| Req ID | Requirement | Description |
|--------|-------------|-------------|
| FR-UI-01 | Responsive Design | UI shall be responsive across desktop, tablet, and mobile |
| FR-UI-02 | Dark Theme | UI shall use a modern dark glassmorphism design |
| FR-UI-03 | Loading States | System shall show animated loading indicators during prediction |
| FR-UI-04 | Toast Notifications | System shall display success/error notifications as toasts |
| FR-UI-05 | Smooth Navigation | Links shall use smooth scroll transitions |
| FR-UI-06 | Scroll Animations | Content shall reveal with intersection observer animations |
| FR-UI-07 | Micro-interactions | Buttons, cards, and icons shall have hover/click animations |
| FR-UI-08 | Modal System | Auth forms shall be presented in animated modal overlays |
| FR-UI-09 | Tab Navigation | Results shall be organized in switchable tab panels |

---

## 4. External Interface Requirements

### 4.1 User Interface (Frontend)

```
Technology: HTML5, CSS3 (Vanilla), JavaScript (ES6+)
Design System: Custom CSS with CSS variables (Design Tokens)
Icons: Lucide Icons (CDN)
Fonts: Inter, JetBrains Mono (Google Fonts)
Theme: Dark glassmorphism with gradient accents
Features:
  - Animated mesh background with grid overlay
  - Glassmorphic cards with backdrop-filter blur
  - Gradient text, gradient buttons
  - Responsive grid layouts
  - Modal overlays for authentication
  - Chip/tag-based symptom selection
  - Tabbed result panels
  - Toast notification system
  - Scroll-reveal animations (IntersectionObserver)
```

### 4.2 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/health` | No | System health check & status |
| `GET` | `/api/symptoms` | No | List all 132 symptoms with severity |
| `GET` | `/api/diseases` | No | List all 41 predictable diseases |
| `POST` | `/api/predict` | Optional | Predict disease from symptoms |
| `POST` | `/api/auth/register` | No | Register new user account |
| `POST` | `/api/auth/login` | No | Login and receive JWT token |
| `POST` | `/api/auth/logout` | No | Logout and clear session |
| `GET` | `/api/auth/me` | Yes | Get current user profile |
| `GET` | `/api/predictions/history` | Yes | Get user's prediction history |
| `POST` | `/api/contact` | No | Submit contact form message |

#### Meta Input Format (POST `/api/predict`)

```json
{
  "symptoms": ["headache", "high_fever", "cough", "fatigue"],
  "patient_info": {
    "age": 30,
    "gender": "male",
    "timestamp": "2026-04-20T12:00:00Z"
  }
}
```

#### Meta Output Format

```json
{
  "success": true,
  "prediction": {
    "disease": "Typhoid",
    "confidence": 93.5,
    "severity_score": 65.2,
    "description": "An acute illness characterized by fever...",
    "precautions": ["eat high calorie vegitables", "antiboitic therapy", "consult doctor", "medication"],
    "medications": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"],
    "diets": ["High calorie diet", "Fluid intake", "Fresh fruits"],
    "workouts": ["Light walking", "Stretching", "Yoga"]
  },
  "input": {
    "symptoms": ["headache", "high_fever", "cough", "fatigue"],
    "invalid_symptoms": [],
    "patient_info": { "age": 30, "gender": "male" }
  },
  "meta": {
    "model": "Support Vector Machine (Linear Kernel)",
    "model_version": "1.0.0",
    "symptoms_analyzed": 4,
    "total_symptoms_available": 132,
    "total_diseases": 41,
    "timestamp": "2026-04-20T12:00:01.234567"
  }
}
```

### 4.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Processor** | Dual-core 2.0 GHz | Quad-core 3.0 GHz |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 500 MB free | 1 GB free |
| **Network** | Internet access for CDN | Stable broadband |

### 4.4 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Flask | 2.3.3 | Web framework |
| MongoDB | 6.0+ | Database |
| scikit-learn | 1.3.0 | ML model |
| NumPy | 1.24.3 | Numerical computing |
| Pandas | 2.0.3 | Data manipulation |
| PyMongo | 4.6.1 | MongoDB driver |
| python-dotenv | 1.0.0 | Environment variables |
| flask-cors | 4.0.0 | CORS handling |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Requirement |
|--------|-------------|
| **Prediction Latency** | < 2 seconds from symptom submission to result |
| **Page Load** | < 3 seconds on 3G connection |
| **API Response** | < 500ms for non-prediction endpoints |
| **Concurrent Users** | Support 50+ simultaneous users |
| **Model Load** | < 5 seconds at server startup |

### 5.2 Security

| Aspect | Implementation |
|--------|---------------|
| **Password Hashing** | PBKDF2-SHA256 with 100,000 iterations + random 16-byte salt |
| **Token Auth** | HMAC-SHA256 signed JWT tokens with expiry |
| **Input Validation** | All API inputs validated and sanitized |
| **CORS** | Configured Cross-Origin Resource Sharing |
| **Session Security** | Server-side session management with secret key |
| **Data Privacy** | Health data encrypted in transit, stored securely |

### 5.3 Reliability & Availability

- **Fallback Mode**: System operates with in-memory storage if MongoDB is unavailable
- **Error Handling**: Graceful error responses with appropriate HTTP status codes
- **Model Integrity**: Pre-trained model loaded at startup with validation
- **Data Validation**: Invalid symptoms filtered with user notification

### 5.4 Usability

- Intuitive symptom search with real-time filtering
- Voice input for accessibility
- Clear visual feedback (toasts, loading states, animations)
- Responsive design for all device sizes
- Severity badges help users prioritize symptoms

### 5.5 Scalability

- Stateless API design allows horizontal scaling
- MongoDB supports sharding for large datasets
- Model can be retrained with additional data using `retrain.py`
- Modular architecture supports feature extension

---

## 6. System Architecture

### 6.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Browser (HTML/CSS/JS)                                │    │
│  │  ├── Glassmorphism UI Theme                           │    │
│  │  ├── Symptom Search + Chip Selector                   │    │
│  │  ├── Web Speech API (Voice Input)                     │    │
│  │  ├── JWT Token Management (localStorage)              │    │
│  │  ├── Fetch API → REST Endpoints                       │    │
│  │  └── IntersectionObserver Animations                  │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/JSON
┌─────────────────────▼───────────────────────────────────────┐
│                     SERVER LAYER (Flask)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Auth Module   │  │ Predict API  │  │ Contact Module   │   │
│  │ ├─ Register   │  │ ├─ /symptoms │  │ └─ /api/contact  │   │
│  │ ├─ Login      │  │ ├─ /predict  │  └──────────────────┘   │
│  │ ├─ Logout     │  │ ├─ /diseases │                         │
│  │ └─ JWT Verify │  │ └─ /history  │                         │
│  └──────┬───────┘  └──────┬───────┘                         │
│         │                  │                                  │
│  ┌──────▼──────────────────▼─────────────────────────────┐   │
│  │              ML ENGINE (scikit-learn)                   │   │
│  │  ├── SVM Model (svc.pkl) — 132 features, 41 classes   │   │
│  │  ├── Feature Vector Builder (binary encoding)          │   │
│  │  ├── Severity Score Calculator                         │   │
│  │  └── Dataset Lookup (descriptions, meds, diets, etc.)  │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │ PyMongo
┌─────────────────────▼───────────────────────────────────────┐
│                   DATABASE LAYER (MongoDB)                    │
│  ┌───────────────┐ ┌─────────────────┐ ┌────────────────┐   │
│  │ users          │ │ predictions      │ │ contacts       │   │
│  │ ├─ name        │ │ ├─ user_id       │ │ ├─ name        │   │
│  │ ├─ email (idx) │ │ ├─ symptoms      │ │ ├─ email       │   │
│  │ ├─ password    │ │ ├─ disease       │ │ ├─ message     │   │
│  │ ├─ avatar      │ │ ├─ confidence    │ │ └─ created_at  │   │
│  │ └─ created_at  │ │ └─ created_at    │ └────────────────┘   │
│  └───────────────┘ └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Data Flow Diagram

```
                    ┌──────────┐
                    │   User   │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │  Select Symptoms    │
              │  (Search / Voice)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  POST /api/predict  │
              │  JSON: {symptoms}   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Build 132-dim      │
              │  Binary Vector      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  SVM Classification │
              │  svc.predict()      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Lookup Disease     │
              │  Data from CSVs     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Return JSON with   │
              │  prediction + meta  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Display Results    │
              │  (Tabs + Cards)     │
              └─────────────────────┘
```

---

# Synopsis Document

## Project Title & Abstract

**Title:** Personalized Medical Recommendation System with Machine Learning

**Abstract:**

The Personalized Medical Recommendation System is an intelligent web-based health application that leverages Machine Learning to predict diseases based on user-reported symptoms. Built with a Flask backend integrated with MongoDB and a modern glassmorphism frontend, the system uses a Support Vector Machine (SVM) classifier trained on 4,920 medical cases to predict 41 different diseases from 132 symptoms. Upon prediction, the system provides comprehensive health recommendations including disease descriptions, precautionary measures, medication suggestions, dietary guidelines, and exercise plans. The application features secure JWT-based authentication, RESTful API design with structured meta input/output format, voice-based symptom input via Web Speech API, prediction history tracking, and an immersive user experience with micro-animations and scroll-reveal effects. This system aims to bridge the gap between symptom onset and preliminary medical guidance, empowering users to make informed health decisions.

---

## Problem Statement

In the current healthcare landscape, patients often face challenges in obtaining quick preliminary health assessments:

1. **Long Wait Times**: Visiting a doctor for every minor symptom is time-consuming and expensive
2. **Symptom Confusion**: People often struggle to connect multiple symptoms to a specific disease
3. **Lack of Awareness**: Many individuals are unaware of precautions, medications, or lifestyle changes for common diseases
4. **Limited Access**: In rural/underserved areas, access to healthcare professionals is limited
5. **Information Overload**: Internet searches for symptoms often lead to anxiety-inducing, inaccurate results

There is a critical need for a **reliable, AI-powered system** that can analyze symptoms and provide structured, actionable health recommendations — acting as a first line of health guidance before professional consultation.

---

## Objectives

### Primary Objectives

1. **Build an ML-based disease prediction engine** using SVM classification on a dataset of 4,920 medical records spanning 41 diseases and 132 symptoms
2. **Develop a full-stack web application** with Flask backend, MongoDB database, and modern responsive frontend
3. **Implement secure user authentication** using JWT tokens and PBKDF2 password hashing
4. **Provide comprehensive health recommendations** including descriptions, precautions, medications, diets, and workouts for each predicted disease
5. **Design RESTful API endpoints** with structured meta input/output format for scalable integration

### Secondary Objectives

6. **Implement voice-based symptom input** using Web Speech API for accessibility
7. **Track prediction history** per user for longitudinal health monitoring
8. **Create premium UI/UX** with glassmorphism design, micro-animations, and scroll-reveal effects
9. **Ensure system reliability** with MongoDB fallback to in-memory storage
10. **Generate severity scores** calculated from symptom weights for risk assessment

---

## Scope

### In Scope

| Area | Details |
|------|---------|
| **Disease Prediction** | Predict 41 diseases from 132 symptom combinations |
| **Health Recommendations** | Medications, diets, workouts, precautions |
| **User Management** | Registration, login, profile, prediction history |
| **API System** | RESTful endpoints with JSON meta format |
| **Frontend UI** | Responsive, animated, dark theme web interface |
| **Database** | MongoDB integration for persistent storage |
| **Voice Input** | Web Speech API for spoken symptom entry |
| **Model Retraining** | Script to retrain SVM with updated data |

### Out of Scope

- Real-time doctor consultation or telemedicine
- Prescription generation or pharmacy integration
- Medical imaging analysis (X-ray, MRI, CT scan)
- Insurance or billing management
- FDA-approved diagnostic certification
- Multi-language support (English only in v2.0)

---

## Methodology

### 1. Data Collection & Preprocessing

- **Dataset**: Medical dataset with 4,920 rows, 132 symptom columns, and 1 target column (prognosis)
- **Supplementary Data**: 6 additional CSV files for descriptions, precautions, medications, diets, workouts, and symptom severity
- **Preprocessing**: Label encoding for target variable, binary encoding for symptom features

### 2. Model Training

```python
Algorithm: Support Vector Machine (SVM)
Kernel: Linear
Train/Test Split: 70/30
Training Samples: 3,444
Test Samples: 1,476
Features: 132 (binary)
Classes: 41 diseases
Encoding: LabelEncoder for target
```

### 3. Backend Development

- Flask web framework with Jinja2 templating
- RESTful API design with JSON responses
- JWT authentication with HMAC-SHA256 signing
- PBKDF2-SHA256 password hashing (100,000 iterations)
- MongoDB integration via PyMongo
- CORS middleware for cross-origin access
- Environment-based configuration via python-dotenv

### 4. Frontend Development

- Vanilla HTML/CSS/JavaScript (no framework dependency)
- CSS Custom Properties (design tokens) for theming
- Glassmorphism design with backdrop-filter
- IntersectionObserver for scroll-reveal animations
- Fetch API for async API communication
- Web Speech API for voice recognition
- Responsive CSS Grid/Flexbox layouts

### 5. Testing & Validation

- API endpoint testing with Python urllib
- ML model accuracy validation (train/test split)
- Cross-browser compatibility testing
- Responsive design testing across viewports
- Authentication flow verification

---

## Technologies Used

### Backend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **Flask** | 2.3.3 | Web framework & API server |
| **scikit-learn** | 1.3.0 | Machine learning (SVM model) |
| **NumPy** | 1.24.3 | Numerical computations |
| **Pandas** | 2.0.3 | Data manipulation & CSV handling |
| **PyMongo** | 4.6.1 | MongoDB driver |
| **python-dotenv** | 1.0.0 | Environment variable management |
| **flask-cors** | 4.0.0 | Cross-Origin Resource Sharing |
| **pickle** | stdlib | Model serialization |

### Frontend Stack

| Technology | Purpose |
|------------|---------|
| **HTML5** | Semantic structure |
| **CSS3** | Glassmorphism styling, animations, CSS Grid |
| **JavaScript (ES6+)** | Application logic, DOM manipulation |
| **Lucide Icons** | Lightweight SVG icon library |
| **Google Fonts** | Inter, JetBrains Mono typography |
| **Web Speech API** | Browser-native voice recognition |

### Database & Infrastructure

| Technology | Purpose |
|------------|---------|
| **MongoDB** | NoSQL document database for users, predictions, contacts |
| **Git** | Version control |
| **pip** | Python package manager |

---

## Expected Outcomes

1. **Accurate Disease Prediction**: ~95% accuracy in predicting diseases from symptom patterns
2. **Comprehensive Health Reports**: Complete recommendation package per prediction (5 categories)
3. **Secure User Platform**: Authentication system protecting user health data
4. **Scalable API**: JSON-based API suitable for mobile app or third-party integration
5. **Accessible Interface**: Voice input and responsive design for broad user accessibility
6. **Persistent History**: Users can track their prediction history over time
7. **Real-time Analysis**: Predictions completed in under 2 seconds

---

## Project Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1**: Research & Planning | Week 1-2 | Literature review, requirements gathering, SRS creation |
| **Phase 2**: Data Preparation | Week 3 | Dataset collection, cleaning, feature engineering |
| **Phase 3**: ML Model Development | Week 4-5 | SVM training, evaluation, model serialization |
| **Phase 4**: Backend Development | Week 5-7 | Flask API, MongoDB integration, authentication |
| **Phase 5**: Frontend Development | Week 7-9 | UI design, animations, responsive layout |
| **Phase 6**: Integration & Testing | Week 9-10 | Full-stack integration, API testing, bug fixing |
| **Phase 7**: Documentation | Week 10-11 | SRS, Synopsis, README, API documentation |
| **Phase 8**: Deployment | Week 11-12 | Final testing, production deployment |

---

## Conclusion

The **Personalized Medical Recommendation System with Machine Learning** successfully demonstrates the application of AI in healthcare informatics. By combining a robust SVM classifier with a modern, secure web platform, the system provides a reliable first-line health assessment tool. The integration of MongoDB for data persistence, JWT for security, and voice input for accessibility ensures the system meets real-world usability requirements. While not a replacement for professional medical advice, Health SymptomSense serves as an empowering tool for preliminary health analysis, particularly valuable in underserved regions and for health-conscious individuals seeking immediate, AI-driven insights.

---

# Project Architecture

## Directory Structure

```
📁 DSML/
├── 📄 main.py                      # Flask application entry point
├── 📄 retrain.py                    # Script to retrain SVM model
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env                          # Environment variables (secrets)
├── 📄 .gitignore                    # Git ignore rules
├── 📄 README.md                     # This documentation file
│
├── 📁 datasets/                     # Training & reference data
│   ├── Training.csv                 # Main training data (4920 × 133)
│   ├── Symptom-severity.csv         # Symptom severity weights
│   ├── symtoms_df.csv               # Symptom-disease mapping
│   ├── description.csv              # Disease descriptions
│   ├── precautions_df.csv           # Disease precautions
│   ├── medications.csv              # Medication recommendations
│   ├── diets.csv                    # Dietary recommendations
│   └── workout_df.csv              # Workout recommendations
│
├── 📁 models/                       # Trained ML models
│   └── svc .pkl                     # Trained SVM model (pickle)
│
├── 📁 templates/                    # Jinja2 HTML templates
│   ├── index.html                   # Main application page
│   ├── about.html                   # About page
│   ├── contact.html                 # Contact page
│   ├── developer.html               # Developer info page
│   └── blog.html                    # Blog page
│
├── 📁 static/                       # Static assets
│   ├── logo.webp                    # Application logo
│   └── *.jpeg                       # Images
│
└── 📄 Medicine Recommendation System.ipynb  # Jupyter notebook
```

---

# Installation & Setup

## Prerequisites

- Python 3.10 or higher
- MongoDB 6.0+ (Community or Atlas)
- pip (Python package manager)
- Git (optional)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DSML.git
cd DSML
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Edit `.env` file with your settings:

```env
# Flask
SECRET_KEY=your-super-secret-key-change-in-production
FLASK_DEBUG=1

# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=health_symptomsense

# JWT
JWT_SECRET_KEY=jwt-secret-key-change-in-production
JWT_ACCESS_TOKEN_EXPIRES=86400

# Server
HOST=0.0.0.0
PORT=5000
```

### 5. Start MongoDB

```bash
# Windows (if installed as service)
net start MongoDB

# Or start mongod manually
mongod --dbpath /data/db
```

### 6. Run the Application

```bash
python main.py
```

The application starts at: **http://localhost:5000**

> **Note**: If MongoDB is not available, the system automatically falls back to in-memory storage mode.

### 7. (Optional) Retrain the Model

```bash
python retrain.py
```

---

# API Documentation

## Base URL

```
http://localhost:5000/api
```

## Authentication

Protected endpoints require a JWT token in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "symptoms_available": 132,
  "diseases_available": 41,
  "version": "2.0.0"
}
```

### List Symptoms

```http
GET /api/symptoms
```

**Response:** Returns array of 132 symptoms with `id`, `key`, `name`, and `severity`.

### List Diseases

```http
GET /api/diseases
```

**Response:** Returns array of 41 diseases with `id` and `name`.

### Predict Disease

```http
POST /api/predict
Content-Type: application/json
Authorization: Bearer <token>  (optional)
```

**Request Body:**
```json
{
  "symptoms": ["headache", "high_fever", "cough"],
  "patient_info": { "age": 25 }
}
```

**Response:** Full prediction with disease, confidence, severity, medications, diets, workouts, precautions, and meta data.

### Register User

```http
POST /api/auth/register
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

### Login User

```http
POST /api/auth/login
Content-Type: application/json
```

### Get Prediction History

```http
GET /api/predictions/history
Authorization: Bearer <token>
```

---

# Database Schema

## MongoDB Collections

### `users` Collection

```json
{
  "_id": "ObjectId",
  "name": "String",
  "email": "String (unique index)",
  "password": "String (PBKDF2 hash)",
  "avatar": "String (first letter of name)",
  "created_at": "DateTime",
  "predictions_count": "Number"
}
```

### `predictions` Collection

```json
{
  "_id": "ObjectId",
  "user_id": "String (ref: users)",
  "symptoms": ["String"],
  "disease": "String",
  "confidence": "Number",
  "severity_score": "Number",
  "patient_info": "Object",
  "created_at": "DateTime"
}
```

### `contacts` Collection

```json
{
  "_id": "ObjectId",
  "name": "String",
  "email": "String",
  "message": "String",
  "created_at": "DateTime"
}
```

---

# Machine Learning Model

## Algorithm: Support Vector Machine (SVM)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Algorithm** | SVC (Support Vector Classification) |
| **Kernel** | Linear |
| **Dataset Size** | 4,920 samples |
| **Features** | 132 (binary symptom indicators) |
| **Classes** | 41 diseases |
| **Train/Test Split** | 70/30 |
| **Random State** | 20 |

### Diseases Predicted (41)

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Fungal infection | 22 | Hepatitis E |
| 2 | Allergy | 23 | Alcoholic hepatitis |
| 3 | GERD | 24 | Tuberculosis |
| 4 | Chronic cholestasis | 25 | Common Cold |
| 5 | Drug Reaction | 26 | Pneumonia |
| 6 | Peptic ulcer disease | 27 | Hemorrhoids (Piles) |
| 7 | AIDS | 28 | Heart attack |
| 8 | Diabetes | 29 | Varicose veins |
| 9 | Gastroenteritis | 30 | Hypothyroidism |
| 10 | Bronchial Asthma | 31 | Hyperthyroidism |
| 11 | Hypertension | 32 | Hypoglycemia |
| 12 | Migraine | 33 | Osteoarthritis |
| 13 | Cervical spondylosis | 34 | Arthritis |
| 14 | Paralysis (brain hemorrhage) | 35 | Vertigo (BPPV) |
| 15 | Jaundice | 36 | Acne |
| 16 | Malaria | 37 | Urinary tract infection |
| 17 | Chicken pox | 38 | Psoriasis |
| 18 | Dengue | 39 | Impetigo |
| 19 | Typhoid | 40 | hepatitis A |
| 20 | Hepatitis B | 41 | Hepatitis D |
| 21 | Hepatitis C | | |

### Feature Engineering

- Each symptom is represented as a binary feature (0 or 1)
- Input vector: 132-dimensional binary array
- Symptom severity weights (1-7 scale) used for severity score calculation

### Model Performance

- **Training Accuracy**: ~100% (linear SVM on separable classes)
- **Test Accuracy**: ~95%
- **Model File**: `models/svc .pkl` (389 KB, pickle serialized)

---

# Screenshots

### Hero Section - Modern Dark Glassmorphism Theme
The landing page features an animated mesh background, gradient text, floating particles, and a responsive hero with real-time stats showing 132 symptoms, 41 diseases, 95% accuracy, and <2s prediction speed.

### Symptom Selector
Interactive search-and-select interface with real-time filtering, severity badges, removable chips, and voice input button with pulsating animation.

### Prediction Results
Tabbed result panels showing disease name with confidence metrics, description, precautions, medications, diets, and workout recommendations in glass-morphic cards.

### Authentication Modal
Sleek overlay modal with login/registration forms, animated error states (shake animation), input focus glow effects, and smooth form transitions.

---

# Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code style
- Use meaningful commit messages
- Add docstrings to new functions
- Test API endpoints before submitting PR
- Update README if adding new features

---

# License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Health SymptomSense Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  <strong>⚕️ Disclaimer</strong>: This system is for informational and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
</p>

<p align="center">
  Built with ❤️ using Python, Flask, scikit-learn, and MongoDB
</p>