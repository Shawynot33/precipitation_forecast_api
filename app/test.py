from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


app = FastAPI()


# Load the models

@app.get("/")
def read_root():
    return {
        "Project" : "Weather Forecast",
        "Objectives" : "Predict precipitation sum of next 3 days / raining or not next 7th day.",
        "endpoints" : {

            "/": "Project overview, endpoints, input/output formats, and GitHub repo",
            "/health/": "Returns status 200 and a welcome message",
            "/predict/rain/": "Predicts whether it will rain in exactly 7 days",
            "/predict/precipitation/fall/": "Predicts cumulative precipitation in the next 3 days"
        },
        
        "predict_rain_input": {
            "date": "string (YYYY-MM-DD)"
        },
        "predict_rain_output": {
            "input_date": "string",
            "prediction": {
                "date": "string (YYYY-MM-DD)",
                "will_rain": "boolean"
            }
        },
        
        "predict_precip_input": {
            "date": "string (YYYY-MM-DD)"
        },
        "predict_precip_output": {
            "input_date": "string",
            "prediction": {
                "start_date": "string (YYYY-MM-DD)",
                "end_date": "string (YYYY-MM-DD)",
                "precipitation_fall": "float (mm)"
            }
        },
        
        "github_repo": "https://github.com/Shawynot33/adv_mla_at2"
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Weather prediciton is all ready to go!'

# Load model once at startup

from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import pandas as pd
import requests
import joblib

model_reg = load("models/xgb_model_reg.joblib")

from datetime import datetime, timedelta
import pandas as pd
from fastapi import Query

@app.get("/predict/precipitation/fall/")
def predict_precipitation(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    try:
        user_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Simulated data covering 10 days
        dates = pd.date_range(user_date - timedelta(days=7), user_date, freq="D")
        df = pd.DataFrame({
            "time": dates.strftime("%Y-%m-%d"),
            "precipitation_sum": [0.5, 1.2, 0.0, 2.1, 3.0, 0.7, 0.0, 1.8],
            "precipitation_hours": [1, 3, 0, 5, 6, 2, 0, 4],
            "cloudcover_mean": [30, 45, 20, 80, 60, 50, 10, 70],
            "vapour_pressure_deficit_max": [1.1, 1.5, 0.8, 2.0, 1.7, 1.3, 0.5, 1.9],
            "shortwave_radiation_sum": [12, 10, 15, 5, 8, 9, 20, 7],
            "wind_gusts_10m_min": [3, 5, 2, 6, 4, 5, 1, 7],
            "wind_direction_10m_dominant": [90, 120, 100, 180, 200, 150, 80, 170],
        })

        # Fill missing values
        df = df.fillna(0)

        # Lag features
        df["precip_sum_lag1"] = df["precipitation_sum"].shift(1).fillna(0)
        df["precip_sum_lag2"] = df["precipitation_sum"].shift(2).fillna(0)

        if str(user_date) not in df["time"].values:
            return {"error": f"Date {user_date} not found in data"}

        features_row = df[df["time"] == str(user_date)].iloc[0]

        feature_columns = [
            "precipitation_sum",
            "precipitation_hours",
            "cloudcover_mean",
            "vapour_pressure_deficit_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "wind_gusts_10m_min",
            "precip_sum_lag1",
            "precip_sum_lag2"
        ]

        X_pred = features_row[feature_columns].to_frame().T
        print("DEBUG - Features sent to model:\n", X_pred)

        # Fake prediction for now
        predicted_precip = 2.34  

        return {
            "input_date": str(user_date),
            "prediction": {
                "start_date": str(user_date + timedelta(days=1)),
                "end_date": str(user_date + timedelta(days=3)),
                "precipitation_fall": str(round(float(predicted_precip), 2))
            }
        }

    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    except Exception as e:
        return {"error": str(e)}

