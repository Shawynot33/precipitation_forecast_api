## Import Packages
# Standard library imports
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
import requests
import requests_cache
from fastapi import FastAPI, Query
from starlette.responses import JSONResponse

# Local or custom modules
import openmeteo_requests
from retry_requests import retry
import joblib
from joblib import load

app = FastAPI()

ARCHIVE_START = datetime(1940, 1, 7).date()
FORECAST_END = datetime.today().date() - timedelta(days=1)


# Functions
def validate_date(date_str: str) -> datetime.date:
    """Validate date string and ensure it's within allowed range."""
    try:
        user_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.")
    
    if user_date < ARCHIVE_START or user_date > FORECAST_END:
        raise ValueError(f"Date {user_date} out of range. Choose between {ARCHIVE_START} and {FORECAST_END}")
    
    return user_date

def fetch_openmeteo_data(url: str, params: dict) -> pd.DataFrame:
    """Fetch data from Open-Meteo API and return as DataFrame."""
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()
    
    daily_data = {}
    for i, var in enumerate(params["daily"]):
        try:
            daily_data[var] = daily.Variables(i).ValuesAsNumpy()
        except AttributeError:  
            daily_data[var] = daily.Variables(i).ValuesInt64AsNumpy()
    
    daily_data["time"] = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    
    df = pd.DataFrame(daily_data)
    df.fillna(0, inplace=True)
    return df

def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Ensure specific columns are numeric."""
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def create_precip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for precipitation model."""
    df["precip_sum_lag1"] = df["precipitation_sum"].shift(1).fillna(0)
    df["precip_sum_lag2"] = df["precipitation_sum"].shift(2).fillna(0)
    return df

def create_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for rain classifier."""
    df['sunrise_dt'] = pd.to_datetime(df['sunrise'], unit='s')
    df['sunset_dt'] = pd.to_datetime(df['sunset'], unit='s')
    df['sunrise_hour'] = df['sunrise_dt'].dt.hour + df['sunrise_dt'].dt.minute/60
    df['sunset_hour'] = df['sunset_dt'].dt.hour + df['sunset_dt'].dt.minute/60
    df['precip_7day_sum'] = df['precipitation_sum'].rolling(7).sum().shift(1)
    df['rh_7day_mean'] = df['relative_humidity_2m_mean'].rolling(7).mean().shift(1)
    df["month_sin"] = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["time"].dt.month / 12)
    return df

def predict_from_model(df: pd.DataFrame, features: list, model, user_date: datetime.date):
    df["date_only"] = df["time"].dt.date
    if user_date not in df["date_only"].values:
        raise ValueError(f"Date {user_date} not found in API data.")
    
    features_row = df[df["date_only"] == user_date].iloc[0]
    X_pred = features_row[features].values.reshape(1, -1)
    return model.predict(X_pred)[0]

@app.get("/")
def read_root():
    return {
        "Project" : "Weather Forecast",
        "Objectives" : "Predict the precipitation sum over the next 3 days to support weather-dependent decision-making, and forecast whether it will rain on the 7th day to help plan operations and minimise weather-related risks.",
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

@app.get("/predict/precipitation/fall/")
def predict_precipitation_endpoint(date: str = Query(...)):
    try:
        user_date = validate_date(date)
        params = {
            "latitude": -33.8678,
            "longitude": 151.2073,
            "start_date": str(user_date - timedelta(days=6)),
            "end_date": str(user_date + timedelta(days=1)),
            "daily": [
                "precipitation_sum",
                "precipitation_hours",
                "cloud_cover_mean",
                "vapour_pressure_deficit_max",
                "shortwave_radiation_sum",
                "wind_gusts_10m_min",
                "wind_direction_10m_dominant"
            ],
            "timezone": "Australia/Sydney"
        }
        df = fetch_openmeteo_data("https://archive-api.open-meteo.com/v1/archive", params)
        df = ensure_numeric(df, params["daily"])
        df = create_precip_features(df)
        numeric_cols = params["daily"] + ["precip_sum_lag1", "precip_sum_lag2"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        predicted_precip = predict_from_model(df, numeric_cols, model_reg, user_date)
        predicted_precip = max(0, predicted_precip)
        return {
            "input_date": str(user_date),
            "prediction": {
                "start_date": str(user_date + timedelta(days=1)),
                "end_date": str(user_date + timedelta(days=3)),
                "precipitation_fall": round(float(predicted_precip), 2)
            }
        }
    except Exception as e:
        return {"error": str(e)}
    
RAIN_FEATURES = [
    'precip_7day_sum',
    'daylight_duration',
    'month_sin',
    'month_cos',
    'sunrise_hour',
    'sunset_hour',
    'rh_7day_mean',
    'apparent_temperature_mean',
    'year'
]


@app.get("/predict/rain/")
def predict_rain_endpoint(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    try:
        # Validate date
        user_date = validate_date(date)

        # Open-Meteo API parameters
        params = {
            "latitude": -33.8678,
            "longitude": 151.2073,
            "start_date": str(user_date - timedelta(days=6)),
            "end_date": str(user_date + timedelta(days=1)),
            "daily": [
                "precipitation_sum",
                "daylight_duration",
                "relative_humidity_2m_mean",
                "apparent_temperature_mean",
                "sunrise",
                "sunset"
            ],
            "timezone": "Australia/Sydney"
        }

        # Fetch and prepare data
        df = fetch_openmeteo_data("https://archive-api.open-meteo.com/v1/archive", params)
        df = ensure_numeric(df, params["daily"])
        df = create_rain_features(df)
        df['year'] = df['time'].dt.year
        df[RAIN_FEATURES] = df[RAIN_FEATURES].astype(float)

        # Predict using model
        predicted_class = predict_from_model(df, RAIN_FEATURES, model_clf, user_date)
        predicted_label = str(bool(int(predicted_class))).upper() # All Caps

        return {
            "input_date": str(user_date),
            "prediction": {
                "date": str(user_date + timedelta(days=7)),
                "will_rain": predicted_label
            }
        }

    except Exception as e:
        return {"error": str(e)}


    
        










    
        








