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

## Modelling Approach
 
Both models were trained using **XGBoost** — chosen for its strong performance on tabular data and flexibility for both regression and classification tasks.
 
**Training pipeline:**
- Time-series split (N=5) applied to preserve temporal ordering across folds, ensuring validation sets simulate future unseen data
- Hyperparameter tuning via **Hyperopt** (Bayesian optimisation, 50 iterations), minimising average validation loss across all folds
- Best parameters used to refit on the full training set, with final evaluation on a held-out **2024 test set**
 
The workflow diagram below illustrates the process:
 
![Modelling Workflow](assets/precip_diagram.png)
 
 
## Model Performance
 
### Regression: 3-Day Precipitation Sum
 
| Metric | Validation | Test | Baseline |
|--------|------------|------|----------|
| RMSE   | 14.15      | 13.98 | 14.83   |
| MAE    | 7.25       | 8.18  | 8.97    |
| R²     | —          | 0.10  | -0.01   |
 
> Precipitation forecasting is inherently high-variance. RMSE and MAE improvements over the mean baseline are the most meaningful indicators of model utility here.
 
### Classification: 7-Day Rain Occurrence
 
| Metric       | Validation | Test  | Baseline |
|--------------|------------|-------|----------|
| F1-score     | 0.725      | 0.660 | 0.763    |
| Weighted F1  | —          | 0.630 | 0.471    |
| Accuracy     | —          | 0.603 | 0.617    |
| ROC-AUC      | 0.635      | 0.668 | 0.500    |
 
> ROC-AUC of 0.668 vs. a 0.500 random baseline reflects meaningful discriminative ability. F1 is sensitive to class imbalance in rain occurrence data and threshold selection.

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shawynot33/precipitation_forecast_api/
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
Displays a brief description of the project objectives, lists available endpoints, expected input parameters, output format, and a link to the [training models repository](https://github.com/Shawynot33/precipitation_forecast).


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
