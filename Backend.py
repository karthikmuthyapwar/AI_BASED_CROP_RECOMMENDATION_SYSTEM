#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask backend for crop recommendation.
"""

import json
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
from tensorflow.keras.preprocessing import image

import soilNET

# initializing flask
app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CROP_DATA_CANDIDATES = [ROOT_DIR / "Cat_Crop.csv", ROOT_DIR / "Cat_Crops.csv"]
PREDICTION_INFO_CANDIDATES = [ROOT_DIR / "Prediction.json"]
CROP_MODEL_CANDIDATES = [
    ROOT_DIR / "finalized_model.sav",
    ROOT_DIR / "Saved Model" / "CRSML.sav",
]


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _error(message, status=400):
    return jsonify({"error": message}), status


@app.route("/", methods=["POST"])
def predict():
    data = request.get_json(force=True)

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
    max_i = np.argmax(prediction)
    if max_i == 0:
        soil = "Alluvial"
    elif max_i == 1:
        soil = "Black"
    elif max_i == 2:
        soil = "Clayey"
    elif max_i == 3:
        soil = "Latterite"
    elif max_i == 4:
        soil = "Red"
    else:
        soil = "Sandy"

    # Restructuring soil type to specific codes according to model input
    soil_mapping = {
        "Alluvial": 1,
        "Black": 2,
        "Clayey": 3,
        "Latterite": 4,
        "Red": 5,
        "Sandy": 6,
    }
    soil_type = soil_mapping.get(soil)

    coordinates = str(data["Loc_Cordinates"])
    locator = Nominatim(user_agent="maati_backend")
    location = locator.reverse(coordinates)
    if location is None or "address" not in location.raw:
        return _error("Could not resolve state from coordinates.", status=422)
    state = location.raw.get("address", {}).get("state")

    state_code = 0
    if state == "Andhra Pradesh":
        state_code = 1
    elif state == "Arunachal Pradesh":
        state_code = 2
    elif state == "Assam":
        state_code = 3
    elif state == "Bihar":
        state_code = 4
    elif state == "Chhatisgarh":
        state_code = 5
    elif state == "Goa":
        state_code = 6
    elif state == "Gujarat":
        state_code = 7
    elif state == "Haryana":
        state_code = 8
    elif state == "Himachal Pradesh":
        state_code = 9
    elif state == "Jharkhand":
        state_code = 10
    elif state == "Karnataka":
        state_code = 11
    elif state == "Kerela":
        state_code = 12
    elif state == "Madhya Pradesh":
        state_code = 13
    elif state == "Maharashtra":
        state_code = 14
    elif state == "Manipur":
        state_code = 15
    elif state == "Meghalaya":
        state_code = 16
    elif state == "Mizoram":
        state_code = 17
    elif state == "Nagaland":
        state_code = 18
    elif state == "Odisha":
        state_code = 19
    elif state == "Punjab":
        state_code = 20
    elif state == "Rajasthan":
        state_code = 21
    elif state == "Sikkim":
        state_code = 22
    elif state == "Tamil Nadu":
        state_code = 23
    elif state == "Telangana":
        state_code = 24
    elif state == "Tripura":
        state_code = 25
    elif state == "Uttar Pradesh":
        state_code = 26
    elif state == "Uttarakhand":
        state_code = 27
    elif state == "West Bengal":
        state_code = 28
    elif state == "Andaman and Nicobar Island":
        state_code = 29
    elif state == "Dadra Nagar Haveli and Daman and Diu":
        state_code = 30
    elif state == "Chandigarh":
        state_code = 31
    elif state == "Delhi":
        state_code = 32
    elif state == "Jammu and Kashmir":
        state_code = 33
    elif state == "Lakshadweep":
        state_code = 34
    elif state == "Pudducherry":
        state_code = 35
    elif state == "Ladakh":
        state_code = 36

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

    temp_values = list(input_dict.values())
    inp_array = np.array(temp_values).reshape(1, -1)
    prediction = loaded_model.predict(inp_array)
    pred_crop_name = prediction[0]

    prediction_info_file = _first_existing(PREDICTION_INFO_CANDIDATES)
    if prediction_info_file is None:
        return _error(
            f"Prediction metadata not found. Expected: {', '.join(str(p) for p in PREDICTION_INFO_CANDIDATES)}",
            status=500,
        )

    with prediction_info_file.open() as fp:
        final_rec = json.load(fp)

    final_pred = final_rec.get(pred_crop_name)
    if final_pred is None:
        return _error(f"No crop metadata found for prediction '{pred_crop_name}'.", status=422)

    final_dict = {"Data": final_pred}
    return json.dumps(final_dict)


if __name__ == "__main__":
    app.run()
