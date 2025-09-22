from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


app = FastAPI()


# Load the models
rain_clf = load('models/xgb_model_clf.joblib')
rain_reg = load('models/xgb_model_reg.joblib')


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
