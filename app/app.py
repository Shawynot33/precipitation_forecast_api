# Import Packages
import streamlit as st
import requests
from datetime import datetime, date, timedelta

API_URL = "http://127.0.0.1:8000"  

st.title("🌦️ Weather Forecast App")

# Set the allowed date range
min_date = date(1940, 1, 7)
max_date = datetime.today().date() - timedelta(days=1)  # yesterday

# Calendar input with restricted range
user_date = st.date_input(
    "Pick a date",
    value=max_date,        # default to yesterday
    min_value=min_date,    # earliest selectable date
    max_value=max_date     # latest selectable date
)


if st.button("Predict 3-Day Precipitation"):
    if user_date:
        resp = requests.get(f"{API_URL}/predict/precipitation/fall/", params={"date": str(user_date)})

        if resp.status_code == 200:
            try:
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
                st.write("Raw response:", resp.text)
        else:
            st.error(f"Error {resp.status_code}")
            st.write("Raw response:", resp.text)
    else:
        st.error("Invalid date format. Please use YYYY-MM-DD.")

if st.button("Predict Rain in 7 Days"):
    if user_date:
        resp = requests.get(f"{API_URL}/predict/rain/", params={"date": str(user_date)})

        if resp.status_code == 200:
            try:
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
                st.write("Raw response:", resp.text)
        else:
            st.error(f"Error {resp.status_code}")
            st.write("Raw response:", resp.text)
    else:
        st.error("Invalid date format. Please use YYYY-MM-DD.")



