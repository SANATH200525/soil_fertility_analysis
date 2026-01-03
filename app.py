from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Soil Fertility API")

# Load model at startup
model_bundle = joblib.load("model/soil_fertility_model.joblib")
model = model_bundle["model"]

@app.get("/")
def home():
    return {"message": "Soil Fertility API is running"}

@app.post("/predict")
def predict_soil(
    ph: float,
    om: float,
    n_no3_ppm: float,
    p_ppm: float,
    k_ppm: float,
    ec_ms_cm: float
):
    """
    Predict soil health category based on soil parameters
    """

    # IMPORTANT: feature order must match training
    input_data = np.array([[ph, om, n_no3_ppm, p_ppm, k_ppm, ec_ms_cm]])

    prediction = model.predict(input_data)[0]

    # Interpret soil health using pH
    if ph < 5.5:
        soil_type = "Acidic"
    elif ph < 7.5:
        soil_type = "Neutral"
    else:
        soil_type = "Alkaline"

    return {
        "soil_type": soil_type,
        "model_output": int(prediction)
    }

