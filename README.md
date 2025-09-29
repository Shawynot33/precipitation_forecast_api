# Precipitation Forecasting API

This repository contains the **API backend** for predicting precipitation using XGBoost models. The API provides forecasts for:  
1. **Precipitation sum over the next three days** (regression)  
2. **Rain occurrence seven days ahead** (classification)  

The models are trained in a separate repository: [Training Models Repository](https://github.com/Shawynot33/adv_mla_at2).


## Features

- Predict precipitation sums (next 3 days)  
- Predict rain occurrence (7 days ahead)  
- Deployed via FastAPI, containerised with Docker, and hosted on **Render**: https://adv-mla-at2-25552249.onrender.com   
- Provides real-time access to predictions for weather-dependent operations  



## Repository Structure
```
├── app/                        
│   ├── app.py           <- FastAPI routes and user interaction  
│   └── main.py          <- Functions handling API logic and responses  
│
├── models/                  
│   ├── xgb_model_clf    <- Saved XGBoost classification model (rain occurrence 7 days ahead)  
│   └── xgb_model_reg    <- Saved XGBoost regression model (precipitation sum next 3 days)  
│
├── .DS_Store  
├── .cache.sqlite  
├── .python-version  
├── Dockerfile           <- Docker configuration for deployment  
├── github.txt  
├── poetry.lock          <- Poetry dependency lock file  
├── pyproject.toml       <- Poetry configuration  
└── requirements.txt     <- Python dependencies  
```

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shawynot33/adv_mla_at2_api/
cd adv_mla_at2_api
```
2. **Install dependencies:**

```bash
# Using Poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

3. **Run the API locally:**
```bash
uvicorn app.main:app --reload
```

## API Endpoints and Usage

The API provides several endpoints to interact with the trained precipitation models. All endpoints are accessed via **GET** requests.


### `/`  (GET)
Displays a brief description of the project objectives, lists available endpoints, expected input parameters, output format, and a link to the [training models repository](https://github.com/Shawynot33/adv_mla_at2).


### `/health/`  (GET)
Returns a status code `200` along with a welcome message.

### `/predict/rain/`  (GET)
Returns the prediction of whether it will rain exactly **7 days** after the input date.  

**Input Parameters:**
- `date`: Date from which the model will predict rain. Format: `YYYY-MM-DD`.  

**Example Request:**
```json
{
  "date": "2023-01-01"
}
```

**Example Response**
```json
{
  "input_date": "2023-01-01",
  "prediction": {
    "date": "2023-01-08",
    "will_rain": true
  }
}
```

### `/predict/precipitation/fall/`  (GET)
Returns the **cumulative precipitation sum** over the next **3 days** from the input date.

**Input Parameters:**
- `date`: Date from which the model will predict precipitation. Format: `YYYY-MM-DD`.

**Example Request:**
```json
{
  "date": "2023-01-01"
}
```

**Example Response:**
```
{
  "input_date": "2023-01-01",
  "prediction": {
    "start_date": "2023-01-02",
    "end_date": "2023-01-04",
    "precipitation_fall": 28.2
  }
}
```
