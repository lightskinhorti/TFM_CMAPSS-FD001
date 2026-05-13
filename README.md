# Pipeline de Mantenimiento Predictivo — NASA CMAPSS

**Máster en Inteligencia Artificial · Digitech · 2026**  
**Autor:** Javier Hortigüela Valiente · [@lightskinhorti](https://github.com/lightskinhorti)

---

## Descripción

Pipeline **end-to-end de Predictive Maintenance** para motores de turbofán sobre el dataset NASA CMAPSS FD001.  
Predice la **Vida Útil Restante (RUL, Remaining Useful Life)** a partir de lecturas de sensores en tiempo real.

```
Datos raw (.txt)
    └── Ingestión + EDA  ·  PySpark
        └── Feature Engineering  ·  Rolling windows (w=5, w=20)
            └── Entrenamiento 5 modelos  ·  MLflow tracking
                └── Evaluación  ·  RMSE + NASA Scoring Function asimétrica
                    └── Despliegue  ·  FastAPI REST endpoint
```

---

## Resultados

| Modelo | RMSE Test | NASA Score Test | Producción |
|---|---|---|---|
| **Ridge α=1.0** | 19.29 | **806.94** | ✅ |
| Random Forest | 18.64 | 855.69 | |
| XGBoost | 19.01 | 1095.80 | |
| MLP (128-64-32) | 19.30 | 1028.67 | |
| Decision Tree | 20.61 | 1044.25 | |

**Hallazgo principal:** Random Forest obtiene el mejor RMSE (18.64) pero un 6% peor NASA Score que Ridge.
XGBoost tiene el segundo mejor RMSE y es un 36% peor en la métrica de negocio.
**La selección del modelo de producción no puede basarse únicamente en el RMSE** cuando el dominio penaliza asimétricamente los errores.

> **NASA Scoring Function** (Saxena et al., 2008): penaliza exponencialmente las predicciones optimistas
> (predecir más vida de la real) más que las pesimistas. Métrica estándar en literatura PHM para RUL.

---

## Stack

| Capa | Tecnología |
|---|---|
| Procesamiento distribuido | Apache PySpark 3.5.3 |
| Feature Engineering | PySpark Window functions particionadas por `unit_id` |
| Modelos | Ridge · Decision Tree · Random Forest · XGBoost · MLP |
| Experiment tracking | MLflow Tracking + Model Registry |
| Despliegue | FastAPI + Pydantic v2 |
| Entorno de desarrollo | Google Colab (Python 3.12) |

---

## Estructura del repositorio

```
TFM_CMAPSS-FD001/
├── TFM_RUL_CMAPSS_FINAL.ipynb          # Notebook principal (ejecutado, outputs incluidos)
├── TFM_Hortiguela_Valiente_FINAL.pdf   # Memoria del TFM (43 páginas)
├── data/
│   ├── train_FD001.txt                 # Trayectorias de entrenamiento (100 motores, hasta fallo)
│   ├── test_FD001.txt                  # Trayectorias de test (truncadas)
│   └── RUL_FD001.txt                   # RUL real del último ciclo de cada motor test
├── api/
│   └── main.py                         # FastAPI — endpoints /health y /predict
├── .gitignore
└── README.md
```

---

## Cómo ejecutar

### Google Colab (recomendado)

1. Abre [`TFM_RUL_CMAPSS_FINAL.ipynb`](TFM_RUL_CMAPSS_FINAL.ipynb) en Colab
2. Los datos se descargan automáticamente desde este repo en la primera celda
3. `Runtime → Run all`

> El notebook ya tiene todos los outputs guardados. No es necesario re-ejecutar para revisar los resultados.

### Local

```bash
git clone https://github.com/lightskinhorti/TFM_CMAPSS-FD001.git
cd TFM_CMAPSS-FD001
pip install pyspark==3.5.3 mlflow scikit-learn xgboost fastapi pydantic uvicorn joblib
jupyter notebook TFM_RUL_CMAPSS_FINAL.ipynb
```

### API de inferencia

```bash
# Genera primero ridge_model.pkl y scaler.pkl ejecutando el notebook
cd api
uvicorn main:app --reload

# Healthcheck
curl http://localhost:8000/health

# Predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 47, "features": [...]}'  # 55 features

# Documentación interactiva
open http://localhost:8000/docs
```

**Niveles de alerta del campo `status`:**

| Status | RUL | Acción |
|---|---|---|
| `critical` | < 30 ciclos | Intervención inmediata |
| `warning` | 30–59 ciclos | Planificar mantenimiento |
| `ok` | ≥ 60 ciclos | Operación normal |

---

## Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) — generado por el Glenn Research Center de la NASA.

- **FD001:** 100 motores · 1 condición operativa · 1 modo de fallo ← *este proyecto*
- FD002: 260 motores · 6 condiciones operativas · 1 modo de fallo
- FD003: 100 motores · 1 condición operativa · 2 modos de fallo
- FD004: 249 motores · 6 condiciones operativas · 2 modos de fallo

Fuente oficial: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

---

## Referencias

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *PHM 2008*. https://doi.org/10.1109/PHM.2008.4711414
- Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *PHM 2008*. https://doi.org/10.1109/PHM.2008.4711422
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*. https://doi.org/10.1145/2939672.2939785
- Zaharia, M. et al. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin, 41*(4), 39–45.
