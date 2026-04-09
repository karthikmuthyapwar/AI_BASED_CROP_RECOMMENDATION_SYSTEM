#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flask backend for crop recommendation + basic user management."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, request
from geopy.geocoders import Nominatim
from tensorflow.keras.preprocessing import image
from werkzeug.security import check_password_hash, generate_password_hash

import soilNET

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "maati.db"

CROP_DATA_CANDIDATES = [
    ROOT_DIR / "Cat_Crop.csv",
    ROOT_DIR / "Cat_Crops.csv",
    ROOT_DIR / "Datasets" / "Cat_Crop.csv",
    ROOT_DIR / "Datasets" / "Cat_Crops.csv",
]
PREDICTION_INFO_CANDIDATES = [
    ROOT_DIR / "Prediction.json",
    ROOT_DIR / "Datasets" / "Prediction.json",
]
CROP_MODEL_CANDIDATES = [
    ROOT_DIR / "finalized_model.sav",
    ROOT_DIR / "Saved Model" / "CRSML.sav",
]

STATE_CODE_MAP = {
    "Andhra Pradesh": 1,
    "Arunachal Pradesh": 2,
    "Assam": 3,
    "Bihar": 4,
    "Chhattisgarh": 5,
    "Chhatisgarh": 5,
    "Goa": 6,
    "Gujarat": 7,
    "Haryana": 8,
    "Himachal Pradesh": 9,
    "Jharkhand": 10,
    "Karnataka": 11,
    "Kerala": 12,
    "Kerela": 12,
    "Madhya Pradesh": 13,
    "Maharashtra": 14,
    "Manipur": 15,
    "Meghalaya": 16,
    "Mizoram": 17,
    "Nagaland": 18,
    "Odisha": 19,
    "Punjab": 20,
    "Rajasthan": 21,
    "Sikkim": 22,
    "Tamil Nadu": 23,
    "Telangana": 24,
    "Tripura": 25,
    "Uttar Pradesh": 26,
    "Uttarakhand": 27,
    "West Bengal": 28,
    "Andaman and Nicobar Islands": 29,
    "Andaman and Nicobar Island": 29,
    "Dadra Nagar Haveli and Daman and Diu": 30,
    "Chandigarh": 31,
    "Delhi": 32,
    "Jammu and Kashmir": 33,
    "Lakshadweep": 34,
    "Puducherry": 35,
    "Pudducherry": 35,
    "Ladakh": 36,
}

SOIL_CODE_MAP = {
    "Alluvial": 1,
    "Black": 2,
    "Clayey": 3,
    "Latterite": 4,
    "Red": 5,
    "Sandy": 6,
}


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _error(message, status=400):
    return jsonify({"error": message}), status


def _get_json() -> dict:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error_obj):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_crop TEXT NOT NULL,
            input_payload TEXT NOT NULL,
            output_payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    db.commit()
    db.close()


def _save_prediction_history(user_id: int, predicted_crop: str, input_payload: dict, output_payload: dict):
    db = get_db()
    db.execute(
        """
        INSERT INTO prediction_history (user_id, predicted_crop, input_payload, output_payload, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            predicted_crop,
            json.dumps(input_payload),
            json.dumps(output_payload),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/auth/register", methods=["POST"])
def register_user():
    data = _get_json()
    name = str(data.get("name", "")).strip()
    phone = str(data.get("phone", "")).strip()
    password = str(data.get("password", "")).strip()

    if not name or not phone or not password:
        return _error("name, phone and password are required.")

    if len(password) < 6:
        return _error("Password must be at least 6 characters long.")

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (name, phone, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, phone, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return _error("User already exists with this phone number.", status=409)

    return jsonify({"message": "User registered successfully."}), 201


@app.route("/auth/login", methods=["POST"])
def login_user():
    data = _get_json()
    phone = str(data.get("phone", "")).strip()
    password = str(data.get("password", "")).strip()

    if not phone or not password:
        return _error("phone and password are required.")

    db = get_db()
    row = db.execute("SELECT id, name, phone, password_hash FROM users WHERE phone = ?", (phone,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return _error("Invalid phone or password.", status=401)

    return jsonify({"user_id": row["id"], "name": row["name"], "phone": row["phone"]})


@app.route("/users/<int:user_id>/history", methods=["GET"])
def prediction_history(user_id: int):
    db = get_db()
    user = db.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        return _error("User not found.", status=404)

    rows = db.execute(
        """
        SELECT predicted_crop, input_payload, output_payload, created_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 50
        """,
        (user_id,),
    ).fetchall()

    history = [
        {
            "predicted_crop": row["predicted_crop"],
            "input": json.loads(row["input_payload"]),
            "output": json.loads(row["output_payload"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return jsonify({"user_id": user_id, "history": history})


@app.route("/", methods=["POST"])
def predict():
    data = _get_json()

    required_keys = ["base64", "ID", "Loc_Cordinates", "Temperature", "date"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        return _error(f"Missing required fields: {missing}")

    base64_img = str(data["base64"])
    file_name = str(data["ID"])
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(base64_img))

    img = image.load_img(file_name)
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    prediction = soilNET.model.predict(x, verbose=0) * 100
    max_i = int(np.argmax(prediction))
    soil = ["Alluvial", "Black", "Clayey", "Latterite", "Red", "Sandy"][max_i]
    soil_type = SOIL_CODE_MAP[soil]

    coordinates = str(data["Loc_Cordinates"])
    locator = Nominatim(user_agent="maati_backend")
    try:
        location = locator.reverse(coordinates)
    except Exception as exc:  # noqa: BLE001
        return _error(f"Location lookup failed: {exc}", status=422)

    if location is None or "address" not in location.raw:
        return _error("Could not resolve state from coordinates.", status=422)

    state = location.raw.get("address", {}).get("state")
    state_code = STATE_CODE_MAP.get(state)
    if state_code is None:
        return _error(f"Unsupported state from coordinates: {state}", status=422)

    crop_data_file = _first_existing(CROP_DATA_CANDIDATES)
    if crop_data_file is None:
        return _error(
            f"Crop dataset not found. Expected one of: {', '.join(str(p) for p in CROP_DATA_CANDIDATES)}",
            status=500,
        )

    file_df = pd.read_csv(crop_data_file)
    data_frame = file_df.loc[file_df["States"] == state_code, "Rainfall"]
    if data_frame.empty:
        return _error("Rainfall data unavailable for resolved state.", status=422)
    rain = float(data_frame.unique()[0])

    df = file_df.loc[file_df["States"] == state_code, "Ground Water"]
    if df.empty:
        return _error("Ground water data unavailable for resolved state.", status=422)
    ground_water = float(df.unique()[0])

    temp = float(data["Temperature"])
    date = str(data["date"])
    month = int(date[5:7])

    if month in [11, 12, 1, 2]:
        season = 2
    elif month in [6, 7, 8, 9]:
        season = 1
    elif month in [3, 4]:
        season = 3
    else:
        season = 4

    input_dict = {
        "States": state_code,
        "Rainfall": rain,
        "Ground Water": ground_water,
        "Temperature": temp,
        "Soil_type": soil_type,
        "Season": season,
    }

    crop_model_file = _first_existing(CROP_MODEL_CANDIDATES)
    if crop_model_file is None:
        return _error(
            f"Crop model file not found. Expected one of: {', '.join(str(p) for p in CROP_MODEL_CANDIDATES)}",
            status=500,
        )

    loaded_model = joblib.load(crop_model_file)
    inp_array = np.array(list(input_dict.values())).reshape(1, -1)
    pred_crop_name = loaded_model.predict(inp_array)[0]

    prediction_info_file = _first_existing(PREDICTION_INFO_CANDIDATES)
    if prediction_info_file is None:
        return _error(
            f"Prediction metadata not found. Expected: {', '.join(str(p) for p in PREDICTION_INFO_CANDIDATES)}",
            status=500,
        )

    with prediction_info_file.open(encoding="utf-8") as fp:
        final_rec = json.load(fp)

    final_pred = final_rec.get(pred_crop_name)
    if final_pred is None:
        return _error(f"No crop metadata found for prediction '{pred_crop_name}'.", status=422)

    final_dict = {"Data": final_pred}

    user_id = data.get("user_id")
    if user_id is not None:
        try:
            _save_prediction_history(int(user_id), str(pred_crop_name), input_dict, final_dict)
        except Exception:
            # Do not fail crop prediction if history persistence fails.
            pass

    return jsonify(final_dict)


if __name__ == "__main__":
    init_db()
    app.run()
