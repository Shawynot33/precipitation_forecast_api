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
git clone <this_repo_url>
cd <this_repo_folder>
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
