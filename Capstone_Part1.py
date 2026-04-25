from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from typing import List, Dict

ML_SERVICE_URL = ""
ACCESS_TOKEN = ""

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

app = FastAPI(title="Retail Forecasting Service")

class ForecastRequest(BaseModel):
    date: str
    product_id: str
    category: str
    region: str
    price: float
    discount: float
    holiday_flag: int

def check_data_drift(record: ForecastRequest) -> str:
    messages = []

    if record.price > 100000:
        messages.append("Price spike detected")
    if record.discount > 80:
        messages.append("Unusual discount value")

    return " | ".join(messages) if messages else "Stable"

def compute_metrics(actual_vals: List[float], predicted_vals: List[float]) -> Dict:
    actual_arr = np.array(actual_vals)
    pred_arr = np.array(predicted_vals)

    mae_value = np.mean(np.abs(actual_arr - pred_arr))
    rmse_value = np.sqrt(np.mean((actual_arr - pred_arr) ** 2))

    return {
        "mae": round(mae_value, 2),
        "rmse": round(rmse_value, 2)
    }

@app.get("/")
def index():
    return {"message": "Retail Forecast API is up and running"}

@app.post("/forecast")
def forecast_demand(request: ForecastRequest):
    try:
        request_payload = {
            "Inputs": {
                "input1": [{
                    "Date": request.date,
                    "ProductID": request.product_id,
                    "Category": request.category,
                    "Region": request.region,
                    "Price": request.price,
                    "Discount": request.discount,
                    "Holiday": request.holiday_flag
                }]
            },
            "GlobalParameters": {}
        }

        api_resp = requests.post(
            ML_SERVICE_URL,
            json=request_payload,
            headers=HEADERS
        )

        prediction_output = api_resp.json()

        drift_info = check_data_drift(request)

        return {
            "status": "ok",
            "prediction_result": prediction_output,
            "drift_analysis": drift_info
        }

    except Exception as err:
        return {
            "status": "failed",
            "error": str(err)
        }

@app.post("/metrics")
def evaluate_model(data: Dict):
    """
    Expected input:
    {
        "actual": [...],
        "predicted": [...]
    }
    """
    try:
        actual_list = data.get("actual", [])
        predicted_list = data.get("predicted", [])

        result_metrics = compute_metrics(actual_list, predicted_list)

        return {
            "status": "ok",
            "evaluation": result_metrics
        }

    except Exception as err:
        return {
            "status": "failed",
            "error": str(err)
        }