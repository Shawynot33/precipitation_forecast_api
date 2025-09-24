from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np

from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

import requests
import joblib

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
    return 'Weather prediction is all ready to go!'

# Load model once at startup
model_reg = load("models/xgb_model_reg.joblib")
model_clf = load("models/xgb_model_clf.joblib")

# Setup Open-Meteo client with cache + retries
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


ARCHIVE_START = datetime(1940, 1, 1).date()
FORECAST_END = datetime.today().date() - timedelta(days=1)


@app.get("/predict/precipitation/fall/")
def predict_precipitation(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    try:
        # Parse input date
        user_date = datetime.strptime(date, "%Y-%m-%d").date()

        
        if user_date < ARCHIVE_START or user_date > FORECAST_END:
            return {
                "error": f"Date {user_date} is out of allowed range. "
                         f"Please use a date between {ARCHIVE_START} and {FORECAST_END}."
            }
        
        # Fetch past 7 days up to user_date
        start_date = user_date - timedelta(days=6)
        end_date = user_date + timedelta(days=1)

        # Open-Meteo request parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": -33.8678,
            "longitude": 151.2073,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "daily": [
                "precipitation_sum",
                "precipitation_hours",
                "cloud_cover_mean",
                "vapour_pressure_deficit_max",
                "shortwave_radiation_sum",
                "wind_gusts_10m_min",
                "wind_direction_10m_dominant"
            ],
            "timezone": "Australia/Sydney",
        }

        # Get response
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Extract daily data
        daily = response.Daily()
        daily_data = {
            "time": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "precipitation_sum": daily.Variables(0).ValuesAsNumpy(),
            "precipitation_hours": daily.Variables(1).ValuesAsNumpy(),
            "cloud_cover_mean": daily.Variables(2).ValuesAsNumpy(),
            "vapour_pressure_deficit_max": daily.Variables(3).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(4).ValuesAsNumpy(),
            "wind_gusts_10m_min": daily.Variables(5).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(6).ValuesAsNumpy(),
        }

        df = pd.DataFrame(data=daily_data)

        # Fill missing values
        df = df.fillna(0)

        # Ensure API columns are numeric first
        api_cols = [
            "precipitation_sum",
            "precipitation_hours",
            "cloud_cover_mean",
            "vapour_pressure_deficit_max",
            "shortwave_radiation_sum",
            "wind_gusts_10m_min",
            "wind_direction_10m_dominant"
        ]
        df[api_cols] = df[api_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Now create lag features
        df["precip_sum_lag1"] = df["precipitation_sum"].shift(1).fillna(0)
        df["precip_sum_lag2"] = df["precipitation_sum"].shift(2).fillna(0)

        # Ensure all feature columns are float
        numeric_cols = api_cols + ["precip_sum_lag1", "precip_sum_lag2"]
        df[numeric_cols] = df[numeric_cols].astype(float)


        # Extract date part from timestamp
        df["date_only"] = df["time"].dt.date
        
        # Check if user_date exists
        if user_date not in df["date_only"].values:
            return {"error": f"Date {user_date} not found in API data."}

        # Select the row
        features_row = df[df["date_only"] == user_date].iloc[0]
        
        X_pred = features_row[numeric_cols].values.reshape(1, -1)
        
        # Predict
        predicted_precip = model_reg.predict(X_pred)[0]

        # Prevent negative prediction
        predicted_precip = max(0, predicted_precip) 

        return {
            "input_date": str(user_date),
            "prediction": {
                "start_date": str(user_date + timedelta(days=1)),
                "end_date": str(user_date + timedelta(days=3)),
                "precipitation_fall": str(round(float(predicted_precip), 2))
            }
        }

    except ValueError:
        return {
            "error": (
                f"Invalid date format: '{date}'. "
                "Please provide the date in YYYY-MM-DD format, e.g., 2025-03-18."
            )
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/rain/")
def predict_precipitation(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    try:
        # Parse input date
        user_date = datetime.strptime(date, "%Y-%m-%d").date()

        
        if user_date < ARCHIVE_START or user_date > FORECAST_END:
            return {
                "error": f"Date {user_date} is out of allowed range. "
                         f"Please use a date between {ARCHIVE_START} and {FORECAST_END}."
            }
        
        # Fetch past 7 days up to user_date
        start_date = user_date - timedelta(days=6)
        end_date = user_date + timedelta(days=1)

        # Open-Meteo request parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": -33.8678,
            "longitude": 151.2073,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "daily": [
                "precipitation_sum",
                "daylight_duration",
                "relative_humidity_2m_mean",
                "temperature_2m_max",
                "apparent_temperature_mean",
                "sunrise",
                "sunset"
            ],
            "timezone": "Australia/Sydney",
        }

        # Get response
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Extract daily data
        daily = response.Daily()
        daily_data = {
            "time": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "precipitation_sum": daily.Variables(0).ValuesAsNumpy(),
            "daylight_duration": daily.Variables(1).ValuesAsNumpy(),
            "relative_humidity_2m_mean": daily.Variables(2).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(3).ValuesAsNumpy(),
            "apparent_temperature_mean": daily.Variables(4).ValuesAsNumpy(),
            "sunrise": daily.Variables(5).ValuesInt64AsNumpy(),
            "sunset": daily.Variables(6).ValuesInt64AsNumpy()
        }

        df = pd.DataFrame(data=daily_data)

        # Endure API columns are numeric first
        api_cols = [
            'precipitation_sum',
            'daylight_duration',
            'relative_humidity_2m_mean',
            'temperature_2m_max',
            'apparent_temperature_mean',
            'sunrise',
            'sunset'
        ]
        df[api_cols] = df[api_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Convert Unix timestamp to datetime
        df['sunrise_dt'] = pd.to_datetime(df['sunrise'], unit='s')
        df['sunset_dt'] = pd.to_datetime(df['sunset'], unit='s')

        # Extract hour + minute as decimal
        df['sunrise_hour'] = df['sunrise_dt'].dt.hour + df['sunrise_dt'].dt.minute/60
        df['sunset_hour'] = df['sunset_dt'].dt.hour + df['sunset_dt'].dt.minute/60

        # Precipitation Rolling Sum (7 days)
        df['precip_7day_sum'] = df['precipitation_sum'].rolling(7).sum().shift(1)

        # Humidity Rolling Mean (7 days)
        df['rh_7day_mean'] = df['relative_humidity_2m_mean'].rolling(7).mean().shift(1)

        # Extract year, month, and encode month cyclically
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Drop month column
        df = df.drop(columns=["month", "sunrise", "sunset", "sunrise_dt", "sunset_dt"])

        # Ensure all feature columns are float
        features_list = [
            'precip_7day_sum',
            'daylight_duration',
            'month_sin',
            'month_cos',
            'sunrise_hour',
            'sunset_hour',
            'rh_7day_mean',
            'temperature_2m_max',
            'year',
            'apparent_temperature_mean'
        ]

        df[features_list] = df[features_list].astype(float)

        # Extract date part from timestamp
        df["date_only"] = df["time"].dt.date

        # Check if user_date exists
        if user_date not in df["date_only"].values:
            return {"error": f"Date {user_date} not found in API data."}

        # Select the row
        features_row = df[df["date_only"] == user_date].iloc[0]

        X_pred = features_row[features_list].values.reshape(1, -1)
        X_pred = X_pred.astype(np.float32).reshape(1, -1)

        predicted_class = model_clf.predict(X_pred)[0]

        predicted_label = bool(int(predicted_class))

        return {
            "input_date": str(user_date),
            "prediction": {
                "date": str(user_date + timedelta(days=7)), 
                "will_rain": predicted_label
            }
        }
    except ValueError:
        return {
            "error": (
                f"Invalid date format: '{date}'. "
                "Please provide the date in YYYY-MM-DD format, e.g., 2025-03-18."
            )
        }
    except Exception as e:
        return {"error": str(e)}
