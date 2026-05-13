
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="RUL Prediction API — NASA CMAPSS",
    description="Predice la Vida Útil Restante de motores de turbofán",
    version="1.0.0"
)

# Cargar modelo y scaler al arrancar la API (una sola vez)
model  = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Schema de entrada ---
# 55 features: 11 sensores + 44 rolling (mean5, std5, mean20, std20)
class SensorInput(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=55,
        max_items=55,
        description="55 features: sensores originales + rolling windows"
    )
    unit_id: int = Field(..., description="ID del motor")

# --- Schema de salida ---
class RULResponse(BaseModel):
    unit_id: int
    rul_predicted: float
    status: str  # "critical" si RUL < 30, "warning" si < 60, "ok" si >= 60

@app.get("/health")
def health():
    """Healthcheck — verifica que la API está operativa."""
    return {"status": "ok", "model": "Ridge", "version": "1.0.0"}

@app.post("/predict", response_model=RULResponse)
def predict(data: SensorInput):
    """
    Recibe 55 features de un motor y devuelve el RUL predicho.
    Las features deben estar en el mismo orden que el training:
    sensores originales (11) + mean5 (11) + std5 (11) + mean20 (11) + std20 (11).
    """
    try:
        X = np.array(data.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        rul = float(np.clip(model.predict(X_scaled)[0], 0, 125))

        if rul < 30:
            status = "critical"
        elif rul < 60:
            status = "warning"
        else:
            status = "ok"

        return RULResponse(
            unit_id=data.unit_id,
            rul_predicted=round(rul, 2),
            status=status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
