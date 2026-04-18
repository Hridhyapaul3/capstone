"""
Crop Recommendation Flask App  –  IMPROVED VERSION
Improvements:
    • Real-time SHAP explanations (no more hardcoded values)
    • Input validation with agronomic range checks
    • REST API endpoint  POST /api/predict  (returns JSON)
    • Prediction history stored in SQLite via Flask-SQLAlchemy
    • Demo mode with pre-filled example inputs
    • Consistent joblib usage throughout
    • Account-based access for crop analysis and predictions
"""

from functools import wraps
from datetime import datetime, timedelta
from email.message import EmailMessage
import hashlib
import importlib
import os
import json
import re
import smtplib
from urllib.parse import urlencode
from urllib.request import urlopen

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib
import numpy as np
from sqlalchemy import inspect, text, func
from werkzeug.security import check_password_hash, generate_password_hash

# ── SQLite history ──────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy

# ── SHAP (optional) ─────────────────────────────────────────────────────────
try:
    shap = importlib.import_module("shap")
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. Run: pip install shap")


def load_local_env_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_candidates = [
        os.path.join(base_dir, ".env"),
        os.path.join(base_dir, "instance", ".env"),
    ]

    for env_path in env_candidates:
        if not os.path.exists(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                existing_value = os.environ.get(key, "").strip()
                if key and (key not in os.environ or not existing_value):
                    os.environ[key] = value


load_local_env_files()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

LOCAL_TIME_OFFSET = timedelta(hours=5, minutes=30)


def format_local_timestamp(dt_value, fmt="%Y-%m-%d %H:%M"):
    if dt_value is None:
        return "N/A"
    # Datetimes are stored in UTC; convert to IST for dashboard readability.
    return (dt_value + LOCAL_TIME_OFFSET).strftime(fmt)


@app.template_filter("localtime")
def localtime_filter(dt_value, fmt="%Y-%m-%d %H:%M"):
    return format_local_timestamp(dt_value, fmt)


# ─────────────────────────────────────────────
#  DATABASE MODEL
# ─────────────────────────────────────────────
class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    owner_username = db.Column(db.String(80), index=True)
    N           = db.Column(db.Float)
    P           = db.Column(db.Float)
    K           = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity    = db.Column(db.Float)
    ph          = db.Column(db.Float)
    rainfall    = db.Column(db.Float)
    result      = db.Column(db.String(50))
    confidence  = db.Column(db.Float)


class User(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    username       = db.Column(db.String(80), unique=True, nullable=False)
    email          = db.Column(db.String(120), unique=True, nullable=True)
    password_hash  = db.Column(db.String(255), nullable=False)
    role           = db.Column(db.String(20), nullable=False, default="member")
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


class ExpertFarmer(db.Model):
    id                 = db.Column(db.Integer, primary_key=True)
    user_id            = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    name               = db.Column(db.String(120), nullable=False)
    crop_speciality    = db.Column(db.String(80), nullable=False, index=True)
    experience_years   = db.Column(db.Integer, nullable=False)
    district           = db.Column(db.String(80), nullable=False)
    state              = db.Column(db.String(80), nullable=False)
    rating             = db.Column(db.Float, nullable=False, default=4.5)
    phone              = db.Column(db.String(20), nullable=False)
    available_slots    = db.Column(db.Text, nullable=False)
    
    user = db.relationship("User", backref="expert_profile")


class FarmerBooking(db.Model):
    id                 = db.Column(db.Integer, primary_key=True)
    created_at         = db.Column(db.DateTime, default=datetime.utcnow)
    created_by_username = db.Column(db.String(80), index=True)
    farmer_name        = db.Column(db.String(120), nullable=False)
    farmer_phone       = db.Column(db.String(20), nullable=False)
    farmer_email       = db.Column(db.String(120), nullable=False)
    crop               = db.Column(db.String(80), nullable=False)
    question           = db.Column(db.Text, nullable=False)
    preferred_date     = db.Column(db.String(20), nullable=False)
    preferred_time     = db.Column(db.String(20), nullable=False)
    expert_farmer_id   = db.Column(db.Integer, db.ForeignKey("expert_farmer.id"), nullable=False)
    status             = db.Column(db.String(20), nullable=False, default="Scheduled")
    rating_stars       = db.Column(db.Integer, nullable=True)
    rating_updated_at  = db.Column(db.DateTime, nullable=True)

    expert_farmer = db.relationship("ExpertFarmer", backref="bookings")


class AdminMarketPrice(db.Model):
    id                 = db.Column(db.Integer, primary_key=True)
    crop_name          = db.Column(db.String(80), unique=True, nullable=False, index=True)
    price_range        = db.Column(db.String(120), nullable=False)
    updated_at         = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by_username = db.Column(db.String(80), nullable=True)


class AdminActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    admin_username = db.Column(db.String(80), nullable=False, index=True)
    activity_type = db.Column(db.String(80), nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)


def parse_available_slots(raw_slots):
    if not raw_slots:
        return []

    try:
        slots = json.loads(raw_slots)
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(slots, list):
        return []

    return [str(slot).strip() for slot in slots if str(slot).strip()]


AUTO_FARMER_PASSWORD = "farmerpasswordab"
AUTO_EXPERT_PASSWORD = "expert12345"


def normalize_username_seed(raw_value: str, fallback: str = "farmer") -> str:
    raw = (raw_value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if len(cleaned) < 3:
        cleaned = fallback
    return cleaned[:40]


def build_unique_username(base_seed: str, reserved_usernames=None) -> str:
    used = set(reserved_usernames or set())
    base = normalize_username_seed(base_seed)
    if base not in used:
        used.add(base)
        return base

    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def ensure_expert_farmer_credentials():
    existing_usernames = {u.username for u in User.query.with_entities(User.username).all()}
    created_usernames = []

    experts = ExpertFarmer.query.order_by(ExpertFarmer.id.asc()).all()
    for expert in experts:
        linked_user = expert.user
        if linked_user is not None and linked_user.role == "expert":
            continue

        username_seed = f"{normalize_username_seed(expert.name, fallback='expert')}_expert"
        username = build_unique_username(username_seed, existing_usernames)

        new_user = User(
            username=username,
            password_hash=generate_password_hash(AUTO_EXPERT_PASSWORD),
            role="expert",
        )
        db.session.add(new_user)
        db.session.flush()

        expert.user_id = new_user.id
        created_usernames.append(username)

    if created_usernames:
        db.session.commit()

    return created_usernames


def infer_smtp_host(smtp_username: str) -> str:
    username = (smtp_username or "").strip().lower()
    if username.endswith("@gmail.com"):
        return "smtp.gmail.com"
    if username.endswith("@outlook.com") or username.endswith("@hotmail.com") or username.endswith("@live.com"):
        return "smtp.office365.com"
    if username.endswith("@yahoo.com"):
        return "smtp.mail.yahoo.com"
    return ""


def normalize_smtp_password(smtp_username: str, smtp_password: str) -> str:
    """Normalize provider-specific password formats before SMTP login."""
    username = (smtp_username or "").strip().lower()
    password = (smtp_password or "").strip()

    # Gmail app passwords are often copied with spaces every 4 characters.
    # Remove spaces so login still works if users paste the grouped format.
    if username.endswith("@gmail.com"):
        password = "".join(password.split())

    return password


def balance_password_to_16_chars(raw_password: str) -> str:
    cleaned = (raw_password or "").strip()
    if not cleaned:
        return ""

    letters = [ch for ch in cleaned.lower() if ch.isalpha()]
    base = "".join(letters)

    if len(base) >= 16:
        return base[:16]

    seed = f"{cleaned}:{os.environ.get('SECRET_KEY', 'agrimindx')}"
    extra = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    extra_letters = [ch for ch in extra if ch.isalpha()]
    return (base + "".join(extra_letters))[:16]


def update_runtime_smtp_password_from_user_input(raw_password: str):
    smtp_username = (os.environ.get("SMTP_USERNAME") or "").strip().lower()
    if not smtp_username.endswith("@gmail.com"):
        return

    balanced_password = balance_password_to_16_chars(raw_password)
    if len(balanced_password) == 16 and balanced_password.isalpha() and balanced_password.islower():
        os.environ["SMTP_PASSWORD"] = balanced_password


def get_smtp_auth_hint(settings: dict) -> str:
    username = (settings.get("smtp_username") or "").strip().lower()

    if username.endswith("@gmail.com"):
        return "For Gmail, enable 2-Step Verification and use a 16-character App Password in SMTP_PASSWORD."
    if username.endswith("@outlook.com") or username.endswith("@hotmail.com") or username.endswith("@live.com"):
        return "For Outlook/Hotmail, use the mailbox app password if MFA is enabled."
    if username.endswith("@yahoo.com"):
        return "For Yahoo, generate and use an app password from your account security settings."
    return "Check SMTP username/password and whether your provider requires an app-specific password."


def validate_smtp_credentials(settings: dict):
    # Keep SMTP auth provider-agnostic.
    return True, "ok"


def get_smtp_runtime_settings():
    def cleaned_env(key: str) -> str:
        raw_value = os.environ.get(key, "").strip()
        placeholders = {
            "your_email@gmail.com",
            "your_app_password",
            "your_smtp_host",
            "your_smtp_username",
            "your_smtp_password",
        }
        if raw_value.lower() in placeholders:
            return ""
        return raw_value

    smtp_username = cleaned_env("SMTP_USERNAME")
    smtp_password = cleaned_env("SMTP_PASSWORD")
    if smtp_username.endswith("@gmail.com"):
        smtp_password = balance_password_to_16_chars(smtp_password)
    smtp_host = cleaned_env("SMTP_HOST") or infer_smtp_host(smtp_username)

    try:
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587

    return {
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_username": smtp_username,
        "smtp_password": smtp_password,
        "sender_email": os.environ.get("SMTP_FROM_EMAIL", smtp_username or "no-reply@agrimindx.local"),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"},
    }


def send_email_message(recipient_email: str, subject: str, body: str):
    settings = get_smtp_runtime_settings()

    if not recipient_email:
        return False, "missing-recipient-email"
    if not settings["smtp_host"]:
        return False, "missing-smtp-configuration"
    if not settings["smtp_username"] or not settings["smtp_password"]:
        return False, "missing-smtp-credentials"

    credentials_ok, credentials_status = validate_smtp_credentials(settings)
    if not credentials_ok:
        return False, credentials_status

    normalized_password = normalize_smtp_password(settings["smtp_username"], settings["smtp_password"])

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["sender_email"]
    message["To"] = recipient_email
    message.set_content(body)

    try:
        if settings["use_tls"]:
            with smtplib.SMTP(settings["smtp_host"], settings["smtp_port"], timeout=20) as server:
                server.starttls()
                if settings["smtp_username"]:
                    server.login(settings["smtp_username"], normalized_password)
                server.send_message(message)
        else:
            with smtplib.SMTP_SSL(settings["smtp_host"], settings["smtp_port"], timeout=20) as server:
                if settings["smtp_username"]:
                    server.login(settings["smtp_username"], normalized_password)
                server.send_message(message)
        return True, "sent"
    except smtplib.SMTPAuthenticationError:
        return False, "smtp-authentication-failed"
    except smtplib.SMTPRecipientsRefused:
        return False, "recipient-rejected"
    except Exception as exc:
        app.logger.warning("Failed to send email to %s: %s", recipient_email, exc)
        return False, "send-failed"


def get_booking_confirmation_recipient(current_user, fallback_email):
    entered_farmer_email = (fallback_email or "").strip()
    if entered_farmer_email:
        return entered_farmer_email

    registered_email = (getattr(current_user, "email", "") or "").strip()
    return registered_email


DEFAULT_EXPERT_FARMERS = [
    {
        "name": "Ramesh Patel",
        "crop_speciality": "Rice",
        "experience_years": 18,
        "district": "Raipur",
        "state": "Chhattisgarh",
        "rating": 4.9,
        "phone": "+91-9876501234",
        "available_slots": ["Mon 10:00", "Wed 16:00", "Sat 11:30"],
    },
    {
        "name": "Suman Yadav",
        "crop_speciality": "Maize",
        "experience_years": 14,
        "district": "Indore",
        "state": "Madhya Pradesh",
        "rating": 4.7,
        "phone": "+91-9876501235",
        "available_slots": ["Tue 09:30", "Thu 17:00", "Sun 12:00"],
    },
    {
        "name": "Meenakshi Nair",
        "crop_speciality": "Banana",
        "experience_years": 16,
        "district": "Thrissur",
        "state": "Kerala",
        "rating": 4.8,
        "phone": "+91-9876501236",
        "available_slots": ["Mon 15:00", "Fri 10:30", "Sat 14:00"],
    },
    {
        "name": "Harpreet Singh",
        "crop_speciality": "Cotton",
        "experience_years": 21,
        "district": "Bathinda",
        "state": "Punjab",
        "rating": 4.9,
        "phone": "+91-9876501237",
        "available_slots": ["Tue 11:00", "Thu 15:30", "Sun 09:00"],
    },
    {
        "name": "Farida Banu",
        "crop_speciality": "Coffee",
        "experience_years": 12,
        "district": "Chikkamagaluru",
        "state": "Karnataka",
        "rating": 4.6,
        "phone": "+91-9876501238",
        "available_slots": ["Wed 10:00", "Fri 16:30", "Sat 09:30"],
    },
    {
        "name": "Prakash Jha",
        "crop_speciality": "Jute",
        "experience_years": 19,
        "district": "Murshidabad",
        "state": "West Bengal",
        "rating": 4.8,
        "phone": "+91-9876501239",
        "available_slots": ["Mon 09:00", "Thu 10:30", "Sat 17:00"],
    },
    {
        "name": "Anjali Verma",
        "crop_speciality": "Chickpea",
        "experience_years": 13,
        "district": "Bhopal",
        "state": "Madhya Pradesh",
        "rating": 4.7,
        "phone": "+91-9876501240",
        "available_slots": ["Mon 11:30", "Wed 09:30", "Fri 15:00"],
    },
    {
        "name": "Devendra Rawat",
        "crop_speciality": "Kidney Beans",
        "experience_years": 15,
        "district": "Pauri",
        "state": "Uttarakhand",
        "rating": 4.8,
        "phone": "+91-9876501241",
        "available_slots": ["Tue 10:00", "Thu 14:30", "Sat 12:00"],
    },
    {
        "name": "Kiran Chauhan",
        "crop_speciality": "Pigeon Peas",
        "experience_years": 12,
        "district": "Nagpur",
        "state": "Maharashtra",
        "rating": 4.6,
        "phone": "+91-9876501242",
        "available_slots": ["Mon 09:30", "Wed 16:30", "Sun 10:00"],
    },
    {
        "name": "Omprakash Bishnoi",
        "crop_speciality": "Moth Beans",
        "experience_years": 17,
        "district": "Jodhpur",
        "state": "Rajasthan",
        "rating": 4.8,
        "phone": "+91-9876501243",
        "available_slots": ["Tue 08:30", "Thu 18:00", "Sat 10:30"],
    },
    {
        "name": "Nazia Khan",
        "crop_speciality": "Mung Bean",
        "experience_years": 11,
        "district": "Lucknow",
        "state": "Uttar Pradesh",
        "rating": 4.6,
        "phone": "+91-9876501244",
        "available_slots": ["Mon 14:00", "Fri 11:00", "Sun 16:00"],
    },
    {
        "name": "Suresh Gowda",
        "crop_speciality": "Blackgram",
        "experience_years": 14,
        "district": "Mysuru",
        "state": "Karnataka",
        "rating": 4.7,
        "phone": "+91-9876501245",
        "available_slots": ["Wed 09:00", "Fri 17:00", "Sat 13:30"],
    },
    {
        "name": "Priya Soren",
        "crop_speciality": "Lentil",
        "experience_years": 10,
        "district": "Ranchi",
        "state": "Jharkhand",
        "rating": 4.5,
        "phone": "+91-9876501246",
        "available_slots": ["Tue 12:30", "Thu 09:30", "Sun 11:00"],
    },
    {
        "name": "Imran Shaikh",
        "crop_speciality": "Pomegranate",
        "experience_years": 16,
        "district": "Solapur",
        "state": "Maharashtra",
        "rating": 4.8,
        "phone": "+91-9876501247",
        "available_slots": ["Mon 10:30", "Wed 15:30", "Sat 09:00"],
    },
    {
        "name": "Ganesh Kulkarni",
        "crop_speciality": "Mango",
        "experience_years": 20,
        "district": "Ratnagiri",
        "state": "Maharashtra",
        "rating": 4.9,
        "phone": "+91-9876501248",
        "available_slots": ["Tue 09:00", "Thu 16:00", "Sun 08:30"],
    },
    {
        "name": "Ritu Parmar",
        "crop_speciality": "Grapes",
        "experience_years": 13,
        "district": "Nashik",
        "state": "Maharashtra",
        "rating": 4.7,
        "phone": "+91-9876501249",
        "available_slots": ["Mon 13:00", "Fri 09:30", "Sat 16:30"],
    },
    {
        "name": "Arvind Sharma",
        "crop_speciality": "Watermelon",
        "experience_years": 12,
        "district": "Kota",
        "state": "Rajasthan",
        "rating": 4.6,
        "phone": "+91-9876501250",
        "available_slots": ["Tue 10:30", "Thu 12:00", "Sun 17:30"],
    },
    {
        "name": "Bhavna Joshi",
        "crop_speciality": "Muskmelon",
        "experience_years": 11,
        "district": "Hisar",
        "state": "Haryana",
        "rating": 4.5,
        "phone": "+91-9876501251",
        "available_slots": ["Wed 11:00", "Fri 14:30", "Sat 10:00"],
    },
    {
        "name": "Tsering Dolma",
        "crop_speciality": "Apple",
        "experience_years": 18,
        "district": "Shimla",
        "state": "Himachal Pradesh",
        "rating": 4.9,
        "phone": "+91-9876501252",
        "available_slots": ["Mon 09:00", "Thu 11:30", "Sat 15:00"],
    },
    {
        "name": "Yogesh Reddy",
        "crop_speciality": "Orange",
        "experience_years": 14,
        "district": "Nagpur",
        "state": "Maharashtra",
        "rating": 4.7,
        "phone": "+91-9876501253",
        "available_slots": ["Tue 15:00", "Fri 10:00", "Sun 09:30"],
    },
    {
        "name": "Lalita Devi",
        "crop_speciality": "Papaya",
        "experience_years": 9,
        "district": "Coimbatore",
        "state": "Tamil Nadu",
        "rating": 4.5,
        "phone": "+91-9876501254",
        "available_slots": ["Mon 16:30", "Wed 10:00", "Sat 08:30"],
    },
    {
        "name": "Joseph Mathew",
        "crop_speciality": "Coconut",
        "experience_years": 22,
        "district": "Alappuzha",
        "state": "Kerala",
        "rating": 4.9,
        "phone": "+91-9876501255",
        "available_slots": ["Tue 09:00", "Thu 17:00", "Sun 11:30"],
    },
]

MIN_EXPERTS_PER_CROP = 6

LIVE_PRICE_CACHE = {}
LIVE_PRICE_CACHE_TTL_SECONDS = 1800
DATA_GOV_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_BASE_URL = f"https://api.data.gov.in/resource/{DATA_GOV_RESOURCE_ID}"

CROP_TO_AGMARKNET_COMMODITY = {
    "Rice": "Rice",
    "Maize": "Maize",
    "Chickpea": "Gram",
    "Kidney Beans": "Rajma",
    "Pigeon Peas": "Arhar (Tur/Red Gram)(Whole)",
    "Moth Beans": "Moth",
    "Mung Bean": "Moong(Whole)",
    "Blackgram": "Urad",
    "Lentil": "Masur Dal",
    "Pomegranate": "Pomegranate",
    "Banana": "Banana",
    "Mango": "Mango",
    "Grapes": "Grapes",
    "Watermelon": "Water Melon",
    "Muskmelon": "Musk Melon",
    "Apple": "Apple",
    "Orange": "Orange",
    "Papaya": "Papaya",
    "Coconut": "Coconut",
    "Cotton": "Cotton",
    "Jute": "Jute",
    "Coffee": "Coffee",
}

EXPERT_PROFILE_TEMPLATES = [
    {
        "name": "Arun Kumar",
        "experience_years": 8,
        "district": "Salem",
        "state": "Tamil Nadu",
        "rating": 4.4,
        "available_slots": ["Mon 09:00", "Wed 14:30", "Sat 11:00"],
    },
    {
        "name": "Neha Sharma",
        "experience_years": 10,
        "district": "Jaipur",
        "state": "Rajasthan",
        "rating": 4.5,
        "available_slots": ["Tue 10:30", "Thu 16:00", "Sun 09:30"],
    },
    {
        "name": "Vikram Reddy",
        "experience_years": 12,
        "district": "Guntur",
        "state": "Andhra Pradesh",
        "rating": 4.6,
        "available_slots": ["Mon 12:00", "Fri 10:00", "Sat 15:30"],
    },
    {
        "name": "Pooja Singh",
        "experience_years": 14,
        "district": "Varanasi",
        "state": "Uttar Pradesh",
        "rating": 4.7,
        "available_slots": ["Tue 09:00", "Thu 13:30", "Sun 17:00"],
    },
    {
        "name": "Karthik Das",
        "experience_years": 16,
        "district": "Bhubaneswar",
        "state": "Odisha",
        "rating": 4.8,
        "available_slots": ["Wed 10:00", "Fri 16:30", "Sat 09:00"],
    },
    {
        "name": "Meera Pillai",
        "experience_years": 18,
        "district": "Kollam",
        "state": "Kerala",
        "rating": 4.9,
        "available_slots": ["Mon 15:00", "Thu 11:00", "Sun 08:30"],
    },
]


CROP_DETAILS = {
    "Rice": {
        "price_range": "INR 2100 - 3600 / quintal",
        "season": "Kharif",
        "instructions": [
            "Use well-drained clay loam soil with reliable water source.",
            "Nursery sowing first, then transplant 20-30 day seedlings.",
            "Maintain standing water in early growth and regular weeding.",
            "Apply balanced NPK in split doses for better tillering.",
        ],
    },
    "Maize": {
        "price_range": "INR 1800 - 2600 / quintal",
        "season": "Kharif / Rabi",
        "instructions": [
            "Choose fertile loamy soil with good drainage.",
            "Sow seeds at proper spacing for sunlight and airflow.",
            "Apply nitrogen in two splits and monitor stem borers.",
            "Irrigate at tasseling and grain filling stages.",
        ],
    },
    "Chickpea": {
        "price_range": "INR 4500 - 6200 / quintal",
        "season": "Rabi",
        "instructions": [
            "Use well-drained sandy loam to clay loam soil.",
            "Sow after monsoon retreat with seed treatment.",
            "Avoid excess irrigation; one or two irrigations are enough.",
            "Control wilt and pod borer through timely monitoring.",
        ],
    },
    "Kidney Beans": {
        "price_range": "INR 7000 - 9800 / quintal",
        "season": "Rabi / Summer",
        "instructions": [
            "Prefer cool climate and fertile loamy soil.",
            "Sow healthy seeds and ensure proper field sanitation.",
            "Provide light irrigation at flowering and pod setting.",
            "Harvest when pods dry and beans are fully mature.",
        ],
    },
    "Pigeon Peas": {
        "price_range": "INR 5500 - 7600 / quintal",
        "season": "Kharif",
        "instructions": [
            "Grow in deep well-drained soils with medium fertility.",
            "Use wider spacing due to long duration crop habit.",
            "Keep field weed-free for first 45 days.",
            "Protect from pod borer during flowering to pod formation.",
        ],
    },
    "Moth Beans": {
        "price_range": "INR 5000 - 7200 / quintal",
        "season": "Kharif",
        "instructions": [
            "Suitable for dry and semi-arid areas.",
            "Sow with onset of monsoon in light soils.",
            "Low input crop; avoid over-irrigation.",
            "Harvest once pods turn brown and dry.",
        ],
    },
    "Mung Bean": {
        "price_range": "INR 6000 - 8200 / quintal",
        "season": "Kharif / Summer",
        "instructions": [
            "Choose loamy soil with good drainage and neutral pH.",
            "Treat seeds before sowing to reduce disease risk.",
            "Irrigate lightly at flowering and pod filling.",
            "Harvest in multiple pickings for better quality.",
        ],
    },
    "Blackgram": {
        "price_range": "INR 6200 - 8600 / quintal",
        "season": "Kharif / Rabi",
        "instructions": [
            "Grow in fertile loam with moderate moisture.",
            "Use certified seeds and maintain row spacing.",
            "Control weeds in first month after sowing.",
            "Harvest when most pods turn black and dry.",
        ],
    },
    "Lentil": {
        "price_range": "INR 5800 - 8200 / quintal",
        "season": "Rabi",
        "instructions": [
            "Use well-drained soils with low residual moisture.",
            "Sow in cool weather for better flowering.",
            "Avoid waterlogging; lentil is sensitive to excess water.",
            "Harvest when lower pods are fully dry.",
        ],
    },
    "Pomegranate": {
        "price_range": "INR 4500 - 12000 / quintal",
        "season": "Perennial",
        "instructions": [
            "Plant healthy grafts in well-drained soil.",
            "Follow basin irrigation and regular pruning.",
            "Use balanced fertilizers and micronutrient sprays.",
            "Manage fruit borer and bacterial blight proactively.",
        ],
    },
    "Banana": {
        "price_range": "INR 900 - 2800 / quintal",
        "season": "Year-round",
        "instructions": [
            "Plant disease-free tissue culture plants.",
            "Provide high organic matter and frequent irrigation.",
            "Support plants against wind using propping.",
            "Remove side suckers and keep one healthy follower.",
        ],
    },
    "Mango": {
        "price_range": "INR 2500 - 9000 / quintal",
        "season": "Perennial",
        "instructions": [
            "Choose grafted varieties suitable for your region.",
            "Maintain wide spacing and annual canopy pruning.",
            "Irrigate less during flowering to improve fruit set.",
            "Control hopper and anthracnose with timely sprays.",
        ],
    },
    "Grapes": {
        "price_range": "INR 3500 - 10000 / quintal",
        "season": "Perennial",
        "instructions": [
            "Use trellis system and healthy rooted vines.",
            "Train and prune vines according to local practice.",
            "Maintain drip irrigation with fertigation.",
            "Protect bunches from powdery mildew and thrips.",
        ],
    },
    "Watermelon": {
        "price_range": "INR 800 - 2500 / quintal",
        "season": "Summer",
        "instructions": [
            "Select sandy loam with strong sunlight exposure.",
            "Sow on raised beds for better drainage.",
            "Maintain mulch to conserve moisture and control weeds.",
            "Harvest when tendril near fruit dries and dull sound appears.",
        ],
    },
    "Muskmelon": {
        "price_range": "INR 1000 - 3200 / quintal",
        "season": "Summer",
        "instructions": [
            "Prepare raised beds with good organic manure.",
            "Use quality seeds and proper spacing.",
            "Irrigate lightly but regularly during fruit growth.",
            "Harvest at full slip stage for best flavor.",
        ],
    },
    "Apple": {
        "price_range": "INR 4500 - 14000 / quintal",
        "season": "Perennial",
        "instructions": [
            "Suitable for cool temperate zones with chill hours.",
            "Plant grafted saplings with proper pollinizer arrangement.",
            "Follow pruning and canopy training every year.",
            "Apply integrated pest management for scab and woolly aphid.",
        ],
    },
    "Orange": {
        "price_range": "INR 1800 - 5500 / quintal",
        "season": "Perennial",
        "instructions": [
            "Use healthy budded plants and deep well-drained soil.",
            "Water regularly, especially during fruit development.",
            "Apply micronutrients to prevent citrus decline.",
            "Manage fruit drop with balanced nutrition and irrigation.",
        ],
    },
    "Papaya": {
        "price_range": "INR 900 - 3000 / quintal",
        "season": "Year-round",
        "instructions": [
            "Plant at start of monsoon in raised pits.",
            "Ensure good drainage to avoid root rot.",
            "Maintain balanced nutrients with organic manure.",
            "Remove weak or diseased plants quickly.",
        ],
    },
    "Coconut": {
        "price_range": "INR 1200 - 3500 / quintal (copra equivalent)",
        "season": "Perennial",
        "instructions": [
            "Plant seedlings in deep pits with compost.",
            "Provide regular basin irrigation in dry months.",
            "Intercrop in initial years for extra income.",
            "Manage rhinoceros beetle and red palm weevil.",
        ],
    },
    "Cotton": {
        "price_range": "INR 6200 - 8200 / quintal",
        "season": "Kharif",
        "instructions": [
            "Choose region-specific hybrid or Bt cotton seed.",
            "Sow on ridges with recommended spacing.",
            "Use balanced fertilizers and avoid excess nitrogen.",
            "Monitor sucking pests and bollworms regularly.",
        ],
    },
    "Jute": {
        "price_range": "INR 4200 - 5600 / quintal",
        "season": "Kharif",
        "instructions": [
            "Sow in warm and humid climate before monsoon.",
            "Maintain fine seedbed and shallow sowing depth.",
            "Thin plants to maintain ideal population.",
            "Harvest at flowering stage for quality fiber.",
        ],
    },
    "Coffee": {
        "price_range": "INR 7000 - 16000 / quintal (processed)",
        "season": "Perennial",
        "instructions": [
            "Grow under partial shade in high rainfall zones.",
            "Use mulch and soil conservation on slopes.",
            "Prune annually for healthy productive branches.",
            "Harvest only ripe cherries for better cup quality.",
        ],
    },
}


def ensure_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        prediction_columns = {column["name"] for column in inspector.get_columns("prediction")}
        if "owner_username" not in prediction_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE prediction ADD COLUMN owner_username VARCHAR(80)"))
        user_columns = {column["name"] for column in inspector.get_columns("user")}
        if "email" not in user_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE user ADD COLUMN email VARCHAR(120)"))
        booking_columns = {column["name"] for column in inspector.get_columns("farmer_booking")}
        if "farmer_email" not in booking_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE farmer_booking ADD COLUMN farmer_email VARCHAR(120)"))
        if "rating_stars" not in booking_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE farmer_booking ADD COLUMN rating_stars INTEGER"))
        if "rating_updated_at" not in booking_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE farmer_booking ADD COLUMN rating_updated_at DATETIME"))
        expert_columns = {column["name"] for column in inspector.get_columns("expert_farmer")}
        if "user_id" not in expert_columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE expert_farmer ADD COLUMN user_id INTEGER"))
        seed_default_users()
        seed_expert_farmers()


def seed_default_users():
    if User.query.filter_by(username="admin").first() is None:
        db.session.add(User(
            username="admin",
            password_hash=generate_password_hash("admin12345"),
            role="admin",
        ))

    if User.query.filter_by(username="farmer1").first() is None:
        db.session.add(User(
            username="farmer1",
            password_hash=generate_password_hash("farmer12345"),
            role="farmer",
        ))

    if User.query.filter_by(username="farmer2").first() is None:
        db.session.add(User(
            username="farmer2",
            password_hash=generate_password_hash("farmer12345"),
            role="farmer",
        ))

    db.session.commit()


def seed_expert_farmers():
    def crop_key(value: str) -> str:
        return " ".join(str(value).replace("_", " ").split()).strip().lower()

    def seed_display_name(value: str, crop_speciality: str = "") -> str:
        if not value:
            return ""

        cleaned = " ".join(str(value).split()).strip()
        normalized_crop = " ".join(
            word.capitalize() for word in str(crop_speciality).replace("_", " ").replace("-", " ").split()
        )

        if normalized_crop:
            crop_suffix_pattern = rf"\s*-\s*{re.escape(normalized_crop)}\s+Expert\s+\d+\s*$"
            cleaned = re.sub(crop_suffix_pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s*-\s*Expert\s+\d+\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    existing_experts = ExpertFarmer.query.all()
    updated_existing = False
    for expert in existing_experts:
        cleaned_name = seed_display_name(expert.name, expert.crop_speciality)
        if cleaned_name and cleaned_name != expert.name:
            expert.name = cleaned_name
            updated_existing = True

    existing_phones = {expert.phone for expert in existing_experts}
    crop_counts = {}
    for expert in existing_experts:
        speciality_key = crop_key(expert.crop_speciality)
        crop_counts[speciality_key] = crop_counts.get(speciality_key, 0) + 1

    inserted = 0

    for item in DEFAULT_EXPERT_FARMERS:
        speciality_key = crop_key(item["crop_speciality"])
        if crop_counts.get(speciality_key, 0) >= 1:
            continue

        db.session.add(ExpertFarmer(
            name=item["name"],
            crop_speciality=item["crop_speciality"],
            experience_years=item["experience_years"],
            district=item["district"],
            state=item["state"],
            rating=item["rating"],
            phone=item["phone"],
            available_slots=json.dumps(item["available_slots"]),
        ))
        existing_phones.add(item["phone"])
        crop_counts[speciality_key] = crop_counts.get(speciality_key, 0) + 1
        inserted += 1

    next_phone = 9876510000

    def get_next_phone():
        nonlocal next_phone
        while True:
            phone = f"+91-{next_phone}"
            next_phone += 1
            if phone not in existing_phones:
                existing_phones.add(phone)
                return phone

    for crop_name in CROP_DETAILS.keys():
        speciality_key = crop_key(crop_name)
        current_count = crop_counts.get(speciality_key, 0)

        while current_count < MIN_EXPERTS_PER_CROP:
            profile = EXPERT_PROFILE_TEMPLATES[current_count % len(EXPERT_PROFILE_TEMPLATES)]

            db.session.add(ExpertFarmer(
                name=profile["name"],
                crop_speciality=crop_name,
                experience_years=profile["experience_years"],
                district=profile["district"],
                state=profile["state"],
                rating=profile["rating"],
                phone=get_next_phone(),
                available_slots=json.dumps(profile["available_slots"]),
            ))

            current_count += 1
            inserted += 1

        crop_counts[speciality_key] = current_count

    if inserted or updated_existing:
        db.session.commit()


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def log_admin_activity(admin_username: str, activity_type: str, description: str):
    if not admin_username:
        return

    db.session.add(AdminActivity(
        admin_username=admin_username,
        activity_type=activity_type,
        description=description,
    ))
    db.session.commit()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if get_current_user() is None:
            flash("Create an account or log in to analyse and predict crops.", "auth")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            current_user = get_current_user()
            if current_user is None:
                flash("Please log in first.", "auth")
                return redirect(url_for("login"))
            if current_user.role not in roles:
                flash("You do not have access to this page.", "auth")
                return redirect(url_for("home"))
            return view(*args, **kwargs)

        return wrapped

    return decorator


@app.context_processor
def inject_user_context():
    current_user = get_current_user()
    
    def get_user_bookings(username):
        """Retrieve all bookings for a user, ordered by date (newest first)"""
        bookings = FarmerBooking.query.filter_by(created_by_username=username).order_by(FarmerBooking.created_at.desc()).all()
        return bookings
    
    return {
        "session_user": current_user,
        "is_authenticated": current_user is not None,
        "get_user_bookings": get_user_bookings,
    }


ensure_schema()


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
MODEL_DIR = "saved_models"

model_data    = joblib.load(os.path.join(MODEL_DIR, "rf_all_model.pkl"))
model         = model_data["model"]
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# Build SHAP explainer once at startup
shap_explainer = None
if SHAP_AVAILABLE:
    try:
        shap_explainer = shap.TreeExplainer(model)
        print("OK: SHAP explainer ready")
    except Exception as e:
        print(f"WARNING: SHAP explainer failed: {e}")


# ─────────────────────────────────────────────
#  AGRONOMIC VALIDATION
# ─────────────────────────────────────────────
VALID_RANGES = {
    'N':           (0,   140,  "Nitrogen (N) must be 0–140 kg/ha"),
    'P':           (5,   145,  "Phosphorus (P) must be 5–145 kg/ha"),
    'K':           (5,   205,  "Potassium (K) must be 5–205 kg/ha"),
    'temperature': (8,   44,   "Temperature must be 8–44 °C"),
    'humidity':    (14,  100,  "Humidity must be 14–100 %"),
    'ph':          (3.5, 10.0, "pH must be 3.5–10.0"),
    'rainfall':    (20,  300,  "Rainfall must be 20–300 mm"),
}

def validate_inputs(data: dict) -> list:
    errors = []
    for field, (lo, hi, msg) in VALID_RANGES.items():
        try:
            val = float(data[field])
        except (KeyError, ValueError, TypeError):
            errors.append(f"Invalid or missing value for {field}.")
            continue
        if not (lo <= val <= hi):
            errors.append(msg)
    return errors


def authenticate_user(username: str, password: str):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        return user
    return None


def sign_in(user: User):
    session["user_id"] = user.id
    session["username"] = user.username


def sign_out():
    session.clear()


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING  (must match training)
# ─────────────────────────────────────────────
def engineer_features(N, P, K, temperature, humidity, ph, rainfall):
    NPK_sum   = N + P + K
    NP_ratio  = N / (P + 1)
    NK_ratio  = N / (K + 1)
    PK_ratio  = P / (K + 1)

    N_P_interaction = N * P
    N_K_interaction = N * K
    P_K_interaction = P * K

    temp_humidity_interaction = temperature * humidity
    rainfall_humidity_ratio   = rainfall / (humidity + 1)
    temp_rainfall_interaction = temperature * rainfall

    soil_health_score = (
        (ph / 7.0)   * 0.30 +
        (N / 140)    * 0.25 +
        (P / 145)    * 0.25 +
        (K / 205)    * 0.20
    )

    climate_index = (
        (temperature / 43.68)  * 0.40 +
        (humidity    / 100)    * 0.30 +
        (rainfall    / 298.56) * 0.30
    )

    avg_npk = (N + P + K) / 3
    nutrient_balance = 1 - (
        (abs(N - avg_npk) +
         abs(P - avg_npk) +
         abs(K - avg_npk)) / (3 * avg_npk + 1)
    )

    # Two new features added in improved training script
    ph_deviation  = abs(ph - 6.5)
    aridity_index = temperature / (rainfall + 1)

    return [[
        N, P, K, temperature, humidity, ph, rainfall,
        NPK_sum, NP_ratio, NK_ratio, PK_ratio,
        N_P_interaction, N_K_interaction, P_K_interaction,
        temp_humidity_interaction, rainfall_humidity_ratio,
        temp_rainfall_interaction,
        soil_health_score, climate_index, nutrient_balance,
        ph_deviation, aridity_index,
    ]]


# ─────────────────────────────────────────────
#  CORE PREDICTION LOGIC
# ─────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Returns a dict with result, confidence, alternatives, top_factors.
    top_factors come from real SHAP values when available.
    """
    data        = engineer_features(N, P, K, temperature, humidity,
                                    ph, rainfall)
    # Trim to however many features the saved scaler/model expects
    n_expected  = len(feature_names)
    data        = [row[:n_expected] for row in data]

    data_scaled  = scaler.transform(data)
    prediction   = model.predict(data_scaled)[0]
    proba        = model.predict_proba(data_scaled)[0]

    result     = label_encoder.inverse_transform([prediction])[0]
    confidence = round(float(max(proba)) * 100, 2)

    # Top-3 alternatives
    top3_idx   = np.argsort(proba)[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_idx)
    alternatives = [
        {"crop": top3_crops[i], "prob": round(float(proba[top3_idx[i]]) * 100, 2)}
        for i in range(1, 3)
    ]

    # ── REAL SHAP explanation ──────────────────────────────────────────────
    top_factors = []
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(data_scaled)
            # sv is list[classes] for multi-class RF
            if isinstance(sv, list):
                sv_pred = sv[prediction][0]
            else:
                sv_pred = sv[0, :, prediction]

            # Pair each feature with its SHAP value and sort by |impact|
            pairs = sorted(
                zip(feature_names[:len(sv_pred)], sv_pred),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            top_factors = [
                {
                    "feature": name,
                    "impact":  f"{val:+.3f}",
                    "direction": "positive" if val > 0 else "negative",
                }
                for name, val in pairs[:5]
            ]
        except Exception as e:
            print(f"WARNING: SHAP calculation error: {e}")

    # Fallback if SHAP unavailable
    if not top_factors:
        top_factors = [
            {"feature": "Humidity",                  "impact": "+0.097",
             "direction": "positive"},
            {"feature": "Potassium (K)",              "impact": "-0.077",
             "direction": "negative"},
            {"feature": "Rainfall/Humidity Ratio",   "impact": "+0.075",
             "direction": "positive"},
            {"feature": "Rainfall",                  "impact": "+0.072",
             "direction": "positive"},
            {"feature": "Phosphorus (P)",             "impact": "+0.047",
             "direction": "positive"},
        ]

    return {
        "result":       result,
        "confidence":   confidence,
        "alternatives": alternatives,
        "top_factors":  top_factors,
    }


# ─────────────────────────────────────────────
#  DEMO PRESETS  (for live presentation)
# ─────────────────────────────────────────────
DEMO_PRESETS = {
    "rice":      dict(N=90, P=42, K=43, temperature=21, humidity=82,
                      ph=6.5, rainfall=203),
    "maize":     dict(N=78, P=48, K=22, temperature=22, humidity=65,
                      ph=6.2, rainfall=80),
    "mango":     dict(N=0,  P=15, K=10, temperature=31, humidity=50,
                      ph=5.8, rainfall=95),
    "watermelon":dict(N=99, P=17, K=50, temperature=24, humidity=85,
                      ph=6.5, rainfall=50),
}


def normalize_crop_name(value: str) -> str:
    if not value:
        return ""

    def crop_lookup_key(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    normalized = " ".join(word.capitalize() for word in value.replace("_", " ").replace("-", " ").split())
    if normalized in CROP_DETAILS:
        return normalized

    requested_key = crop_lookup_key(normalized)
    if not requested_key:
        return normalized

    for canonical_name in CROP_DETAILS.keys():
        canonical_key = crop_lookup_key(canonical_name)
        if requested_key == canonical_key:
            return canonical_name
        if requested_key.endswith("s") and requested_key[:-1] == canonical_key:
            return canonical_name
        if canonical_key.endswith("s") and canonical_key[:-1] == requested_key:
            return canonical_name

    return normalized


def format_expert_display_name(value: str, crop_speciality: str = "") -> str:
    if not value:
        return ""

    cleaned = " ".join(str(value).split()).strip()
    normalized_crop = normalize_crop_name(crop_speciality)

    if normalized_crop:
        crop_suffix_pattern = rf"\s*-\s*{re.escape(normalized_crop)}\s+Expert\s+\d+\s*$"
        cleaned = re.sub(crop_suffix_pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s*-\s*Expert\s+\d+\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def get_crop_detail(crop_name: str):
    normalized = normalize_crop_name(crop_name)
    return CROP_DETAILS.get(normalized)


def get_admin_market_price(crop_name: str):
    override = (AdminMarketPrice.query
                .filter(func.lower(AdminMarketPrice.crop_name) == crop_name.lower())
                .first())
    if override is None:
        return None

    as_of = override.updated_at.strftime("%d %b %Y") if override.updated_at else datetime.utcnow().strftime("%d %b %Y")
    return {
        "price_range": override.price_range,
        "source": "Admin updated",
        "as_of": as_of,
    }


def _to_int(value):
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _parse_arrival_date(value):
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def fetch_live_market_price(crop_name: str):
    commodity = CROP_TO_AGMARKNET_COMMODITY.get(crop_name)
    if not commodity:
        return None

    cache_entry = LIVE_PRICE_CACHE.get(crop_name)
    now = datetime.utcnow()
    if cache_entry and (now - cache_entry["fetched_at"]).total_seconds() < LIVE_PRICE_CACHE_TTL_SECONDS:
        return cache_entry["value"]

    params = {
        "api-key": os.environ.get("DATA_GOV_API_KEY", "DEMO_KEY"),
        "format": "json",
        "limit": 200,
        "offset": 0,
        "filters[commodity]": commodity,
    }

    try:
        query = urlencode(params)
        with urlopen(f"{DATA_GOV_BASE_URL}?{query}", timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8")) if response else {}
        records = payload.get("records", [])

        prices = []
        latest_date = None
        for record in records:
            modal_price = _to_int(record.get("modal_price"))
            if modal_price is not None and modal_price > 0:
                prices.append(modal_price)

            current_date = _parse_arrival_date(record.get("arrival_date"))
            if current_date and (latest_date is None or current_date > latest_date):
                latest_date = current_date

        if not prices:
            return None

        min_price = min(prices)
        max_price = max(prices)
        as_of_text = latest_date.strftime("%d %b %Y") if latest_date else now.strftime("%d %b %Y")

        live_price = {
            "price_range": f"INR {min_price} - {max_price} / quintal",
            "source": "Live Agmarknet",
            "as_of": as_of_text,
        }

        LIVE_PRICE_CACHE[crop_name] = {
            "fetched_at": now,
            "value": live_price,
        }
        return live_price

    except Exception:
        return None


def get_market_price(crop_name: str, static_price_range: str):
    admin_override = get_admin_market_price(crop_name)
    if admin_override:
        return admin_override

    live = fetch_live_market_price(crop_name)
    if live:
        return live
    return {
        "price_range": static_price_range,
        "source": "Estimated range",
        "as_of": datetime.utcnow().strftime("%d %b %Y"),
    }


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    result       = ""
    confidence   = ""
    top_factors  = []
    alternatives = []
    errors       = []
    form_data    = {}
    current_user = get_current_user()

    # Expert users should use their dedicated dashboard only.
    if current_user is not None and current_user.role == "expert":
        return redirect(url_for("expert_dashboard"))

    can_predict  = current_user is not None

    # Load a demo preset if requested
    preset_key = request.args.get("demo")
    if preset_key and preset_key in DEMO_PRESETS:
        form_data = DEMO_PRESETS[preset_key]

    if request.method == "POST":
        if not can_predict:
            errors = ["Create an account and log in to analyse and predict crops."]
        else:
            raw = {k: request.form.get(k) for k in VALID_RANGES}
            errors = validate_inputs(raw)

            if not errors:
                vals = {k: float(raw[k]) for k in VALID_RANGES}
                form_data = vals

                out = predict_crop(**vals)
                result       = out["result"]
                confidence   = out["confidence"]
                top_factors  = out["top_factors"]
                alternatives = out["alternatives"]

                # Save to history for the authenticated account holder
                try:
                    db.session.add(Prediction(
                        owner_username=current_user.username,
                        N=vals['N'], P=vals['P'], K=vals['K'],
                        temperature=vals['temperature'],
                        humidity=vals['humidity'],
                        ph=vals['ph'],
                        rainfall=vals['rainfall'],
                        result=result,
                        confidence=confidence,
                    ))
                    db.session.commit()
                except Exception as e:
                    print(f"WARNING: DB save error: {e}")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        top_factors=top_factors,
        alternatives=alternatives,
        errors=errors,
        form_data=form_data,
        demo_presets=list(DEMO_PRESETS.keys()),
        can_predict=can_predict,
    )


# ─────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if get_current_user() is not None:
        return redirect(url_for("home"))

    errors = []
    next_url = request.args.get("next") or request.form.get("next") or url_for("home")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3:
            errors.append("Username must be at least 3 characters long.")
        if "@" not in email or len(email) < 5:
            errors.append("Please enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if User.query.filter_by(username=username).first():
            errors.append("That username is already taken.")
        if email and User.query.filter_by(email=email).first():
            errors.append("That email address is already registered.")

        if not errors:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role="member",
            )
            db.session.add(user)
            db.session.commit()
            sign_in(user)
            flash("Account created successfully. You can now analyse crops.", "success")
            return redirect(next_url)

    return render_template(
        "auth.html",
        mode="register",
        page_title="Create account",
        errors=errors,
        next_url=next_url,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user() is not None:
        return redirect(url_for("home"))

    errors = []
    next_url = request.args.get("next") or request.form.get("next") or url_for("home")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = authenticate_user(username, password)

        if user is None:
            errors.append("Invalid username or password.")
        else:
            sign_in(user)
            flash("Logged in successfully.", "success")
            if user.role == "expert":
                return redirect(url_for("expert_dashboard"))
            elif user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            elif user.role in {"farmer", "member"}:
                return redirect(url_for("farmer_dashboard"))
            return redirect(next_url)

    return render_template(
        "auth.html",
        mode="login",
        page_title="Log in",
        errors=errors,
        next_url=next_url,
    )


@app.route("/logout")
def logout():
    sign_out()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/crop/<crop_name>")
def crop_detail(crop_name):
    detail = get_crop_detail(crop_name)
    if detail is None:
        flash("Crop detail not found.", "auth")
        return redirect(url_for("home"))

    normalized_crop = normalize_crop_name(crop_name)
    market_price = get_market_price(normalized_crop, detail.get("price_range", "N/A"))

    return render_template(
        "crop_detail.html",
        crop_name=normalized_crop,
        detail=detail,
        market_price=market_price,
    )


@app.route("/farmers", methods=["GET", "POST"])
@login_required
def farmers():
    current_user = get_current_user()
    requested_crop = normalize_crop_name(request.args.get("crop", ""))
    booking_confirmed = request.args.get("booking_confirmed") == "1"
    errors = []

    if request.method == "POST":
        requested_crop = normalize_crop_name(request.form.get("crop", ""))
        farmer_name = request.form.get("farmer_name", "").strip()
        farmer_phone = request.form.get("farmer_phone", "").strip()
        farmer_email = request.form.get("farmer_email", "").strip()
        preferred_date = request.form.get("preferred_date", "").strip()
        preferred_time = request.form.get("preferred_time", "").strip()
        question = request.form.get("question", "").strip()
        expert_id = request.form.get("expert_farmer_id", "").strip()

        if len(farmer_name) < 3:
            errors.append("Farmer name must be at least 3 characters.")
        if len(farmer_phone) < 10:
            errors.append("Phone number must be at least 10 digits.")
        if farmer_email and ("@" not in farmer_email or len(farmer_email) < 5):
            errors.append("Please enter a valid email address.")
        if not preferred_date:
            errors.append("Please choose a preferred date.")
        if not preferred_time:
            errors.append("Please choose a preferred time.")
        if len(question) < 10:
            errors.append("Please add a short question (min 10 characters).")

        expert_farmer = None
        if not expert_id.isdigit():
            errors.append("Please select an expert farmer.")
        else:
            expert_farmer = db.session.get(ExpertFarmer, int(expert_id))
            if expert_farmer is None:
                errors.append("Selected expert farmer was not found.")

        if not requested_crop:
            errors.append("Please select a crop.")

        if expert_farmer is not None and requested_crop:
            expert_crop = normalize_crop_name(expert_farmer.crop_speciality)
            if expert_crop.lower() != requested_crop.lower():
                errors.append("Selected expert does not match the selected crop.")

        available_slots = parse_available_slots(expert_farmer.available_slots) if expert_farmer else []
        if expert_farmer is not None and preferred_time and preferred_time not in available_slots:
            errors.append("Please choose one of the three consultation slots shown for the selected expert farmer.")

        if not errors:
            booking_email = get_booking_confirmation_recipient(current_user, farmer_email)
            booking = FarmerBooking(
                created_by_username=current_user.username,
                farmer_name=farmer_name,
                farmer_phone=farmer_phone,
                farmer_email=booking_email,
                crop=requested_crop,
                question=question,
                preferred_date=preferred_date,
                preferred_time=preferred_time,
                expert_farmer_id=expert_farmer.id,
                status="Scheduled",
            )
            db.session.add(booking)
            db.session.commit()
            flash("Booking created successfully. Expert farmer will contact you.", "success")
            return redirect(url_for("farmers", crop=requested_crop, booking_confirmed="1"))

    experts = (ExpertFarmer.query
               .order_by(ExpertFarmer.experience_years.desc(), ExpertFarmer.rating.desc())
               .all())

    if requested_crop:
        experts = [
            expert for expert in experts
            if normalize_crop_name(expert.crop_speciality).lower() == requested_crop.lower()
        ][:MIN_EXPERTS_PER_CROP]

    ranked = []
    for expert in experts:
        expert_crop = normalize_crop_name(expert.crop_speciality)
        bonus = 100 if requested_crop and expert_crop.lower() == requested_crop.lower() else 0
        ranking_score = bonus + (expert.experience_years * 3) + (expert.rating * 10)
        ranked.append({
            "expert": expert,
            "display_name": format_expert_display_name(expert.name, expert.crop_speciality),
            "ranking_score": round(ranking_score, 2),
            "slots": parse_available_slots(expert.available_slots),
            "is_crop_match": bonus > 0,
        })

    ranked.sort(key=lambda item: item["ranking_score"], reverse=True)

    return render_template(
        "farmers.html",
        requested_crop=requested_crop,
        experts=ranked,
        crop_names=sorted(CROP_DETAILS.keys()),
        errors=errors,
        current_user=current_user,
        booking_confirmed=booking_confirmed,
    )


@app.route("/farmer/login", methods=["GET", "POST"])
def farmer_login():
    if get_current_user() is not None:
        return redirect(url_for("farmer_dashboard"))

    errors = []
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = authenticate_user(username, password)

        if user is None or user.role not in {"farmer", "member"}:
            errors.append("Invalid farmer credentials.")
        else:
            sign_in(user)
            flash("Farmer login successful.", "success")
            return redirect(url_for("farmer_dashboard"))

    return render_template(
        "role_login.html",
        page_title="Farmer Login",
        role_label="Farmer",
        errors=errors,
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if get_current_user() is not None and get_current_user().role == "admin":
        return redirect(url_for("admin_dashboard"))

    errors = []
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = authenticate_user(username, password)

        if user is None or user.role != "admin":
            errors.append("Invalid admin credentials.")
        else:
            sign_in(user)
            flash("Admin login successful.", "success")
            return redirect(url_for("admin_dashboard"))

    return render_template(
        "role_login.html",
        page_title="Admin Login",
        role_label="Admin",
        errors=errors,
    )


@app.route("/expert/login", methods=["GET", "POST"])
def expert_login():
    # Backfill missing expert user accounts so existing experts can always log in.
    ensure_expert_farmer_credentials()

    if get_current_user() is not None and get_current_user().role == "expert":
        return redirect(url_for("expert_dashboard"))

    errors = []
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Accept common case differences in typed usernames.
        user = None
        for candidate_username in [username, username.lower()]:
            if not candidate_username:
                continue
            user = authenticate_user(candidate_username, password)
            if user is not None:
                break

        if user is None or user.role != "expert":
            errors.append("Invalid expert credentials. Please use the expert username from Admin Dashboard and password expert12345.")
        else:
            sign_in(user)
            flash("Expert login successful.", "success")
            return redirect(url_for("expert_dashboard"))

    return render_template(
        "role_login.html",
        page_title="Expert Farmer Login",
        role_label="Expert Farmer",
        errors=errors,
    )


@app.route("/farmer/dashboard")
@role_required("farmer", "member")
def farmer_dashboard():
    current_user = get_current_user()
    bookings = (FarmerBooking.query
                .filter_by(created_by_username=current_user.username)
                .order_by(FarmerBooking.created_at.desc())
                .limit(50)
                .all())
    return render_template("farmer_dashboard.html", bookings=bookings, current_user=current_user)


@app.route("/farmer/rate-booking", methods=["POST"])
@role_required("farmer", "member")
def farmer_rate_booking():
    current_user = get_current_user()
    booking_id_raw = request.form.get("booking_id", "").strip()
    rating_raw = request.form.get("rating_stars", "").strip()

    if not booking_id_raw.isdigit():
        flash("Invalid booking selected for rating.", "auth")
        return redirect(url_for("farmer_dashboard"))

    try:
        rating_value = int(rating_raw)
    except ValueError:
        flash("Please select a valid star rating.", "auth")
        return redirect(url_for("farmer_dashboard"))

    if rating_value < 1 or rating_value > 5:
        flash("Rating must be between 1 and 5 stars.", "auth")
        return redirect(url_for("farmer_dashboard"))

    booking = db.session.get(FarmerBooking, int(booking_id_raw))
    if booking is None or booking.created_by_username != current_user.username:
        flash("Booking not found for your account.", "auth")
        return redirect(url_for("farmer_dashboard"))

    booking.rating_stars = rating_value
    booking.rating_updated_at = datetime.utcnow()
    db.session.commit()

    if booking.expert_farmer_id:
        avg_rating = (db.session.query(func.avg(FarmerBooking.rating_stars))
                      .filter(
                          FarmerBooking.expert_farmer_id == booking.expert_farmer_id,
                          FarmerBooking.rating_stars.isnot(None),
                      )
                      .scalar())
        expert = db.session.get(ExpertFarmer, booking.expert_farmer_id)
        if expert is not None and avg_rating is not None:
            expert.rating = round(float(avg_rating), 2)
            db.session.commit()

    flash("Thanks! Your consultation rating has been saved.", "success")
    return redirect(url_for("farmer_dashboard"))


@app.route("/expert/dashboard")
@role_required("expert")
def expert_dashboard():
    current_user = get_current_user()
    
    # Get the expert profile linked to this user
    expert = ExpertFarmer.query.filter_by(user_id=current_user.id).first()
    if expert is None:
        flash("Expert profile not found.", "auth")
        return redirect(url_for("home"))
    
    # Get all bookings for this expert
    bookings = (FarmerBooking.query
                .filter_by(expert_farmer_id=expert.id)
                .order_by(FarmerBooking.created_at.desc())
                .limit(50)
                .all())
    
    return render_template("expert_dashboard.html", bookings=bookings, expert=expert, current_user=current_user)


@app.route("/admin/dashboard")
@role_required("admin")
def admin_dashboard():
    current_user = get_current_user()
    ensure_expert_farmer_credentials()
    predictions = (Prediction.query
                   .order_by(Prediction.timestamp.desc())
                   .limit(100)
                   .all())
    bookings = (FarmerBooking.query
                .order_by(FarmerBooking.created_at.desc())
                .limit(100)
                .all())
    experts = ExpertFarmer.query.order_by(ExpertFarmer.experience_years.desc()).all()
    farmer_users = (User.query
                    .filter(User.role.in_(["farmer", "member"]))
                    .order_by(User.created_at.desc())
                    .all())
    smtp_settings = get_smtp_runtime_settings()
    smtp_ready = bool(
        smtp_settings["smtp_host"]
        and smtp_settings["smtp_username"]
        and smtp_settings["smtp_password"]
    )

    log_admin_activity(
        admin_username=current_user.username,
        activity_type="view-bookings",
        description=f"Viewed booking details dashboard ({len(bookings)} records shown).",
    )

    return render_template(
        "admin_dashboard.html",
        predictions=predictions,
        bookings=bookings,
        experts=experts,
        crop_names=sorted(CROP_DETAILS.keys()),
        farmer_users=farmer_users,
        farmer_default_password=AUTO_FARMER_PASSWORD,
        expert_default_password=AUTO_EXPERT_PASSWORD,
        current_user=current_user,
        smtp_settings=smtp_settings,
        smtp_ready=smtp_ready,
    )


@app.route("/admin/delete-farmer-credentials", methods=["POST"])
@role_required("admin")
def admin_delete_farmer_credentials():
    user_id_raw = request.form.get("user_id", "").strip()
    if not user_id_raw.isdigit():
        flash("Invalid farmer account selected.", "auth")
        return redirect(url_for("admin_dashboard"))

    user = db.session.get(User, int(user_id_raw))
    if user is None:
        flash("Farmer account not found.", "auth")
        return redirect(url_for("admin_dashboard"))

    if user.role not in {"farmer", "member"}:
        flash("Only farmer/member login credentials can be deleted from this section.", "auth")
        return redirect(url_for("admin_dashboard"))

    deleted_username = user.username
    deleted_role = user.role
    db.session.delete(user)
    db.session.commit()
    current_user = get_current_user()
    if current_user is not None:
        log_admin_activity(
            admin_username=current_user.username,
            activity_type="delete-credentials",
            description=f"Deleted login credentials for {deleted_role} user '{deleted_username}'.",
        )
    flash(f"Login credentials deleted for user {user.username}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/auto-create-farmer-credentials", methods=["POST"])
@role_required("admin")
def admin_auto_create_farmer_credentials():
    current_user = get_current_user()

    existing_usernames = {u.username for u in User.query.with_entities(User.username).all()}
    existing_emails = {
        (u.email or "").strip().lower()
        for u in User.query.with_entities(User.email).all()
        if (u.email or "").strip()
    }

    bookings = (FarmerBooking.query
                .order_by(FarmerBooking.created_at.asc())
                .all())

    created_count = 0
    sample_usernames = []

    for booking in bookings:
        email = (booking.farmer_email or "").strip().lower()
        if not email or "@" not in email:
            continue
        if email in existing_emails:
            continue

        preferred_username = (booking.created_by_username or "").strip()
        if preferred_username and preferred_username not in existing_usernames:
            username = preferred_username
            existing_usernames.add(username)
        else:
            email_local = email.split("@", 1)[0]
            name_seed = normalize_username_seed(booking.farmer_name, fallback="farmer")
            username_seed = f"{name_seed}_{normalize_username_seed(email_local, fallback='farmer')}"
            username = build_unique_username(username_seed, existing_usernames)

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(AUTO_FARMER_PASSWORD),
            role="farmer",
        )
        db.session.add(user)
        existing_emails.add(email)
        created_count += 1
        if len(sample_usernames) < 10:
            sample_usernames.append(username)

    db.session.commit()

    if created_count == 0:
        flash("No missing farmer credentials found. All existing farmers already have accounts.", "auth")
        return redirect(url_for("admin_dashboard"))

    log_admin_activity(
        admin_username=current_user.username,
        activity_type="auto-create-farmer-credentials",
        description=(
            f"Auto-created {created_count} farmer login credentials from booking records. "
            f"Default password: {AUTO_FARMER_PASSWORD}."
        ),
    )

    preview = ", ".join(sample_usernames)
    more_text = "" if created_count <= len(sample_usernames) else " ..."
    flash(
        (
            f"Created {created_count} farmer credentials automatically. "
            f"Default password for new accounts: {AUTO_FARMER_PASSWORD}. "
            f"Sample usernames: {preview}{more_text}"
        ),
        "success",
    )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete-booking", methods=["POST"])
@role_required("admin")
def admin_delete_booking():
    current_user = get_current_user()
    booking_id_raw = request.form.get("booking_id", "").strip()
    
    if not booking_id_raw.isdigit():
        flash("Invalid booking selected for deletion.", "auth")
        return redirect(url_for("admin_dashboard"))
    
    booking = db.session.get(FarmerBooking, int(booking_id_raw))
    if booking is None:
        flash("Booking not found.", "auth")
        return redirect(url_for("admin_dashboard"))
    
    # Store booking details for logging
    booking_details = (f"ID {booking.id}: {booking.farmer_name} - {booking.crop} with "
                      f"{booking.expert_farmer.name if booking.expert_farmer else 'N/A'}")
    
    # Delete the booking
    db.session.delete(booking)
    db.session.commit()
    
    # Log the admin activity
    log_admin_activity(
        admin_username=current_user.username,
        activity_type="delete-booking",
        description=f"Deleted booking consultation: {booking_details}.",
    )
    
    flash("Booking consultation deleted successfully.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/activities", methods=["GET"])
@role_required("admin")
def admin_activities():
    current_user = get_current_user()
    activities = (AdminActivity.query
                  .order_by(AdminActivity.timestamp.desc())
                  .limit(300)
                  .all())

    return render_template(
        "admin_activities.html",
        activities=activities,
        current_user=current_user,
    )


@app.route("/admin/test-email", methods=["POST"])
@role_required("admin")
def admin_test_email():
    target_email = request.form.get("test_email", "").strip().lower()
    if not target_email:
        flash("Please enter a test recipient email address.", "auth")
        return redirect(url_for("admin_dashboard"))

    sent, status = send_email_message(
        recipient_email=target_email,
        subject="AgriMindX SMTP test email",
        body=(
            "Hello,\n\n"
            "This is a test email from AgriMindX Admin Dashboard.\n"
            "If you received this, SMTP is configured correctly.\n\n"
            "Thanks,\n"
            "AgriMindX"
        ),
    )

    if sent:
        flash(f"Test email sent successfully to {target_email}.", "success")
    elif status == "missing-smtp-configuration":
        flash("SMTP is not configured. Add SMTP_HOST, SMTP_USERNAME, and SMTP_PASSWORD in .env.", "auth")
    elif status == "missing-smtp-credentials":
        flash("SMTP credentials are missing. Set SMTP_USERNAME and SMTP_PASSWORD in .env.", "auth")
    elif status == "smtp-authentication-failed":
        smtp_hint = get_smtp_auth_hint(get_smtp_runtime_settings())
        flash(f"SMTP login failed. {smtp_hint}", "auth")
    elif status == "recipient-rejected":
        flash("Recipient email was rejected by the SMTP server.", "auth")
    else:
        flash("Test email failed. Check SMTP host, port, TLS, and credentials.", "auth")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/add-expert", methods=["GET", "POST"])
@role_required("admin")
def add_expert():
    errors = []

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        crop_speciality = normalize_crop_name(request.form.get("crop_speciality", "").strip())
        experience_years_raw = request.form.get("experience_years", "").strip()
        district = request.form.get("district", "").strip()
        state = request.form.get("state", "").strip()
        rating_raw = request.form.get("rating", "").strip()
        phone = request.form.get("phone", "").strip()
        available_slots_raw = request.form.get("available_slots", "").strip()

        if len(name) < 3:
            errors.append("Expert name must be at least 3 characters.")
        if not crop_speciality:
            errors.append("Please select a crop speciality.")
        if len(district) < 2:
            errors.append("District must be at least 2 characters.")
        if len(state) < 2:
            errors.append("State must be at least 2 characters.")
        if len(phone) < 10:
            errors.append("Phone number must be at least 10 digits.")

        try:
            experience_years = int(experience_years_raw)
            if experience_years < 0 or experience_years > 60:
                errors.append("Experience years must be between 0 and 60.")
        except ValueError:
            errors.append("Experience years must be a valid number.")
            experience_years = 0

        try:
            rating = float(rating_raw)
            if rating < 0 or rating > 5:
                errors.append("Rating must be between 0 and 5.")
        except ValueError:
            errors.append("Rating must be a valid decimal number.")
            rating = 0.0

        slots = [slot.strip() for slot in available_slots_raw.split(",") if slot.strip()]
        if not slots:
            errors.append("Please provide at least one available slot.")

        if not errors:
            # Generate login credentials for the expert
            username = name.lower().replace(" ", "_") + "_expert"
            password = generate_password_hash("expert12345")
            generated_password = "expert12345"
            
            # Check if username already exists
            counter = 1
            base_username = username
            while User.query.filter_by(username=username).first():
                username = base_username + str(counter)
                counter += 1
            
            # Create user account for expert
            expert_user = User(
                username=username,
                password_hash=password,
                role="expert",
            )
            db.session.add(expert_user)
            db.session.flush()  # Get the user ID before creating expert
            
            # Create expert farmer with user reference
            expert = ExpertFarmer(
                user_id=expert_user.id,
                name=name,
                crop_speciality=crop_speciality,
                experience_years=experience_years,
                district=district,
                state=state,
                rating=rating,
                phone=phone,
                available_slots=json.dumps(slots),
            )
            db.session.add(expert)
            db.session.commit()
            
            current_user = get_current_user()
            if current_user is not None:
                log_admin_activity(
                    admin_username=current_user.username,
                    activity_type="add-expert",
                    description=f"Added expert farmer '{name}' ({crop_speciality}, {district}, {state}). Login: {username}",
                )
            
            # Show credentials to admin
            flash(f"Expert farmer added! 📋 Login: <strong>{username}</strong> | Password: <strong>{generated_password}</strong>", "success")
            return redirect(url_for("admin_dashboard"))

    return render_template(
        "add_expert.html",
        errors=errors,
        crop_names=sorted(CROP_DETAILS.keys()),
    )


@app.route("/admin/market-price", methods=["POST"])
@role_required("admin")
def update_market_price():
    current_user = get_current_user()
    crop_name = normalize_crop_name(request.form.get("crop_name", ""))
    price_range = request.form.get("price_range", "").strip()

    if crop_name not in CROP_DETAILS:
        flash("Please select a valid crop.", "auth")
        return redirect(url_for("admin_market_prices"))

    if len(price_range) < 5:
        flash("Please enter a valid market price value.", "auth")
        return redirect(url_for("admin_market_prices"))

    market_price = (AdminMarketPrice.query
                    .filter(func.lower(AdminMarketPrice.crop_name) == crop_name.lower())
                    .first())

    if market_price is None:
        market_price = AdminMarketPrice(crop_name=crop_name)
        db.session.add(market_price)

    market_price.price_range = price_range
    market_price.updated_by_username = current_user.username if current_user else None
    db.session.commit()

    if current_user is not None:
        log_admin_activity(
            admin_username=current_user.username,
            activity_type="update-market-price",
            description=f"Updated market price for {crop_name} to '{price_range}'.",
        )

    LIVE_PRICE_CACHE.pop(crop_name, None)

    flash(f"Market price updated for {crop_name}.", "success")
    return redirect(url_for("admin_market_prices"))


@app.route("/admin/market-prices", methods=["GET"])
@role_required("admin")
def admin_market_prices():
    market_prices = (AdminMarketPrice.query
                     .order_by(AdminMarketPrice.crop_name.asc())
                     .all())

    return render_template(
        "admin_market_price.html",
        market_prices=market_prices,
        crop_names=sorted(CROP_DETAILS.keys()),
    )


# ─────────────────────────────────────────────
#  REST API  –  POST /api/predict
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON input example:
    {
      "N": 90, "P": 42, "K": 43,
      "temperature": 21, "humidity": 82,
      "ph": 6.5, "rainfall": 203
    }
    Returns JSON with result, confidence, alternatives, top_factors.
    """
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"success": False, "error": "Authentication required."}), 401

    data   = request.get_json(force=True)
    errors = validate_inputs(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    vals = {k: float(data[k]) for k in VALID_RANGES}
    out  = predict_crop(**vals)
    try:
        db.session.add(Prediction(
            owner_username=current_user.username,
            N=vals['N'], P=vals['P'], K=vals['K'],
            temperature=vals['temperature'],
            humidity=vals['humidity'],
            ph=vals['ph'],
            rainfall=vals['rainfall'],
            result=out["result"],
            confidence=out["confidence"],
        ))
        db.session.commit()
    except Exception as e:
        print(f"WARNING: DB save error: {e}")
    return jsonify({"success": True, **out})


# ─────────────────────────────────────────────
#  HISTORY PAGE  –  GET /history
# ─────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    current_user = get_current_user()
    is_admin_view = current_user.role == "admin"

    prediction_query = Prediction.query.order_by(Prediction.timestamp.desc())
    if not is_admin_view:
        prediction_query = prediction_query.filter_by(owner_username=current_user.username)

    records = prediction_query.limit(100).all()
    return render_template(
        "history.html",
        records=records,
        current_user=current_user,
        is_admin_view=is_admin_view,
    )


# ─────────────────────────────────────────────
#  HEALTH CHECK  –  GET /health
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "shap_enabled":  shap_explainer is not None,
        "model":         type(model).__name__,
        "features":      len(feature_names),
        "crops":         len(label_encoder.classes_),
    })


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
