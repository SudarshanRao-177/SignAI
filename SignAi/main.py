from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import json
import os
import numpy as np
import joblib


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "sign_model_v5.pkl")
DATASET_FILE = os.path.join(BASE_DIR, "sign_dataset.json")
GESTURES_DIR = os.path.join(BASE_DIR, "gestures")


# =========================================================
# NLTK
# =========================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SignAI Backend",
    version="1.0.0"
)

# GIFs are available at /gestures/<filename>.gif
app.mount(
    "/gestures",
    StaticFiles(directory=GESTURES_DIR),
    name="gestures"
)


# =========================================================
# CORS
# =========================================================
# Local development + deployed frontend origins.
# allow_origins=["*"] is used so the same backend can serve
# the pages locally and on Render without CORS blocking.
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD TRAINED ML MODEL
# =========================================================

model = None

try:

    if not os.path.exists(MODEL_FILE):

        print("========================================")
        print("WARNING: ML MODEL FILE NOT FOUND")
        print("Expected file:")
        print(MODEL_FILE)
        print("========================================")

    else:

        model = joblib.load(MODEL_FILE)

        print("========================================")
        print("ML MODEL LOADED SUCCESSFULLY")
        print("Model:", MODEL_FILE)
        print("========================================")

except Exception as error:

    model = None

    print("========================================")
    print("WARNING: ML MODEL NOT LOADED")
    print("Error:", error)
    print("========================================")


# =========================================================
# GESTURE / GIF MAPPING
# =========================================================

gesture_map = {

    # Main 10 signs
    "hello": "hello",
    "yes": "yes",
    "no": "no",
    "thanks": "thankyou",
    "thank": "thankyou",
    "please": "please",
    "help": "help",
    "sorry": "sorry",
    "stop": "stop",
    "water": "water",
    "pain": "pain",

    # Additional GIFs
    "howru": "howareyou",
    "wtyourname": "whatisyourname",
    "goodmorning": "goodmorning",

    # Common words
    "morning": "goodmorning"
}


# =========================================================
# STOPWORDS
# =========================================================

stop_words = set(stopwords.words("english"))


# =========================================================
# TEXT INPUT
# =========================================================

class TextInput(BaseModel):
    text: str


# =========================================================
# TEXT → GESTURE
# =========================================================

@app.post("/text-to-gesture")
def text_to_gesture(data: TextInput):

    text = data.text.lower().strip()

    text = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in text
    )

    text = " ".join(text.split())

    if not text:

        return {
            "sequence": ["unknown"]
        }

    if text in [
        "how are you",
        "how r you",
        "howru"
    ]:

        return {
            "sequence": ["howareyou"]
        }

    if text == "good morning":

        return {
            "sequence": ["goodmorning"]
        }

    if text == "thank you":

        return {
            "sequence": ["thankyou"]
        }

    if text in [
        "what is your name",
        "whats your name",
        "what s your name",
        "what is ur name",
        "what s ur name",
        "wtyourname"
    ]:

        return {
            "sequence": ["whatisyourname"]
        }

    words = word_tokenize(text)

    result = []

    for word in words:

        if word in gesture_map:

            result.append(
                gesture_map[word]
            )

    if not result:

        result = ["unknown"]

    return {
        "sequence": result
    }


# =========================================================
# DATASET COLLECTION
# =========================================================

class LandmarkInput(BaseModel):
    label: str
    landmarks: list[float]


@app.post("/collect-landmark")
def collect_landmark(data: LandmarkInput):

    dataset = []

    if os.path.exists(DATASET_FILE):

        try:

            with open(DATASET_FILE, "r") as file:
                dataset = json.load(file)

        except Exception:

            dataset = []

    if len(data.landmarks) != 63:

        return {
            "status": "error",
            "message": (
                f"Expected 63 landmarks, "
                f"received {len(data.landmarks)}"
            ),
            "total_samples": len(dataset)
        }

    dataset.append(
        {
            "label": data.label,
            "landmarks": data.landmarks
        }
    )

    with open(DATASET_FILE, "w") as file:

        json.dump(
            dataset,
            file
        )

    return {
        "status": "saved",
        "total_samples": len(dataset)
    }


# =========================================================
# SAVE DATASET BATCH
# =========================================================

class LandmarkBatch(BaseModel):
    samples: list


@app.post("/collect-batch")
def collect_batch(data: LandmarkBatch):

    dataset = []

    if os.path.exists(DATASET_FILE):

        try:

            with open(DATASET_FILE, "r") as file:
                dataset = json.load(file)

        except Exception:

            dataset = []

    dataset.extend(data.samples)

    with open(DATASET_FILE, "w") as file:

        json.dump(
            dataset,
            file
        )

    return {
        "status": "saved",
        "added": len(data.samples),
        "total_samples": len(dataset)
    }


# =========================================================
# NATURAL SENTENCE FORMATION
# =========================================================

class SentenceInput(BaseModel):
    words: list[str]


def build_natural_sentence(words):

    cleaned = []

    for word in words:

        word = str(word).lower().strip()

        if word and word != "unknown":

            cleaned.append(word)

    if not cleaned:

        return ""

    compact = []

    for word in cleaned:

        if not compact or word != compact[-1]:

            compact.append(word)

    key = tuple(compact)

    exact_phrases = {

        ("hello", "help"):
            "Hello, I need help.",

        ("hello", "help", "please"):
            "Hello, I need help, please.",

        ("hello", "help", "please", "water"):
            "Hello, I need help, please give me water.",

        ("hello", "help", "water"):
            "Hello, I need help with water.",

        ("hello", "please", "water"):
            "Hello, please give me water.",

        ("help", "please"):
            "Please help me.",

        ("help", "please", "water"):
            "Please help me, and give me water.",

        ("help", "water"):
            "I need help with water.",

        ("please", "water"):
            "Please give me water.",

        ("sorry", "thanks"):
            "I am sorry, thank you.",

        ("hello", "thanks"):
            "Hello, thank you.",

        ("hello", "sorry"):
            "Hello, I am sorry.",

        ("good", "morning"):
            "Good morning.",

        ("goodmorning",):
            "Good morning.",

        ("howareyou",):
            "How are you?",

        ("whatisyourname",):
            "What is your name?"
    }

    if key in exact_phrases:

        return exact_phrases[key]

    has_hello = "hello" in compact
    has_help = "help" in compact
    has_please = "please" in compact
    has_water = "water" in compact
    has_thanks = "thanks" in compact
    has_sorry = "sorry" in compact
    has_stop = "stop" in compact

    if has_stop:

        return "Please stop."

    if has_hello and has_help and has_please and has_water:

        return "Hello, I need help, please give me water."

    if has_hello and has_help and has_please:

        return "Hello, I need help, please."

    if has_hello and has_help and has_water:

        return "Hello, I need help with water."

    if has_hello and has_please and has_water:

        return "Hello, please give me water."

    if has_help and has_please and has_water:

        return "Please help me, and give me water."

    if has_sorry and has_thanks:

        return "I am sorry, thank you."

    if len(compact) == 1:

        single = compact[0]

        single_map = {

            "hello": "Hello.",
            "help": "I need help.",
            "please": "Please.",
            "water": "I need water.",
            "thankyou": "Thank you.",
            "thanks": "Thank you.",
            "sorry": "I am sorry.",
            "yes": "Yes.",
            "no": "No.",
            "stop": "Please stop.",
            "pain": "I am in pain.",
            "howareyou": "How are you?",
            "whatisyourname": "What is your name?",
            "goodmorning": "Good morning."
        }

        if single in single_map:

            return single_map[single]

    return " ".join(compact).capitalize() + "."


# =========================================================
# NATURAL SENTENCE API
# =========================================================

@app.post("/natural-sentence")
def natural_sentence(data: SentenceInput):

    return {
        "sentence": build_natural_sentence(data.words)
    }


# =========================================================
# REAL-TIME SIGN PREDICTION
# =========================================================

class PredictionInput(BaseModel):
    landmarks: list[float]


@app.post("/predict-sign")
def predict_sign(data: PredictionInput):

    if model is None:

        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "error": "ML model is not loaded"
        }

    if len(data.landmarks) != 63:

        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "error": (
                f"Expected 63 landmarks, "
                f"received {len(data.landmarks)}"
            )
        }

    try:

        features = np.array(
            data.landmarks,
            dtype=float
        ).reshape(1, -1)

        prediction = model.predict(
            features
        )[0]

        confidence = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = float(
                np.max(probabilities)
            )

        return {

            "prediction": str(prediction),

            "confidence": round(
                confidence,
                4
            )
        }

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return {

            "prediction": "unknown",

            "confidence": 0.0,

            "error": str(error)
        }


# =========================================================
# WEBSITE PAGES
# =========================================================
#
# IMPORTANT:
# FastAPI serves the complete frontend from the same server.
# This fixes /index.html -> Not Found and also makes the
# logo, GIFs and other files available on Render.
# =========================================================

@app.get("/")
def root():

    return FileResponse(
        os.path.join(BASE_DIR, "index.html")
    )


@app.get("/index.html")
def index_page():

    return FileResponse(
        os.path.join(BASE_DIR, "index.html")
    )


@app.get("/realtime.html")
def realtime_page():

    return FileResponse(
        os.path.join(BASE_DIR, "realtime.html")
    )


@app.get("/practice.html")
def practice_page():

    return FileResponse(
        os.path.join(BASE_DIR, "practice.html")
    )


@app.get("/collect.html")
def collect_page():

    return FileResponse(
        os.path.join(BASE_DIR, "collect.html")
    )


# =========================================================
# FRONTEND ASSETS
# =========================================================
#
# The HTML files use relative paths such as:
#   SignAI LOGO.png
#   gestures/hello.gif
#
# GIFs are already mounted above. This catch-all route
# serves files such as the logo and any other frontend asset.
#
# It is deliberately placed AFTER all API/page routes.
# =========================================================

@app.get("/{file_path:path}")
def frontend_asset(file_path: str):

    # Never allow ../ path traversal.
    safe_path = os.path.normpath(
        os.path.join(BASE_DIR, file_path)
    )

    if not safe_path.startswith(os.path.abspath(BASE_DIR)):
        return {"detail": "Not Found"}

    if os.path.isfile(safe_path):

        return FileResponse(safe_path)

    return {"detail": "Not Found"}


# =========================================================
# SERVER STARTUP MESSAGE
# =========================================================

@app.on_event("startup")
def startup_message():

    print("")

    print("========================================")
    print("        SIGN AI BACKEND READY")
    print("========================================")

    print("Home             : /")
    print("Index page       : /index.html")
    print("Live Sign        : /realtime.html")
    print("Practice         : /practice.html")
    print("Dataset          : /collect.html")
    print("Text conversion  : /text-to-gesture")
    print("Dataset sample   : /collect-landmark")
    print("Dataset batch    : /collect-batch")
    print("AI prediction    : /predict-sign")
    print("Logo             : /SignAI%20LOGO.png")
    print("Model loaded     :", model is not None)

    print("========================================")

    print("")