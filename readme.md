# Disaster Prediction API

**Disaster Prediction API** is a machine learning-based backend service that predicts the likelihood of natural disasters such as floods, storms, or extreme weather events based on meteorological data. The API is built with Python and FastAPI, making it fast, scalable, and easy to integrate into other applications.  

---

## Features
- Predicts disasters using weather and environmental data.
- Handles multiple input features including:
  - `date`
  - `precipitation`
  - `temp_max`
  - `temp_min`
  - `wind`
- Calculates derived features:
  - Temperature difference (`temp_diff`)
  - Average temperature (`temp_avg`)
- Returns structured predictions in JSON format.
- Easily extensible with new models or features.

---

## Tech Stack
- **Backend Framework:** FastAPI
- **Machine Learning:** CatBoost / XGBoost (configurable)
- **Data Processing:** pandas, numpy
- **Deployment:** Docker-ready, compatible with fly.io, Simple Front-end Design, mlflow on Dagshub

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/disaster-prediction-api.git
cd disaster-prediction-api
```

2. Create a virtual environment:
```bash 
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run app localy 
```bash
uvicorn app:app --reload
```
## Usage
Send a POST request to `/predict` endpoint with JSON body:  
```json
{
  "features": [
    {
      "date": "2025-08-28",
      "precipitation": 12.3,
      "temp_max": 35,
      "temp_min": 28,
      "wind": 5
    }
  ]
}
```
result: 
```json
{
  "predictions": [rain]  
}
```

---

## Contributing
- Fork the repo
- Create a feature branch
- Make your changes and test
- Submit a pull request  

---

## License
MIT License © 2025 Aly El-Deen Yasser Ali

---
links :
- Github : https://github.com/Aly-EL-Badry/DisasterPrediction
- API : https://disasterprediction.fly.dev/predict
- DockerFile: https://hub.docker.com/r/alyelbadry/disaster-prediction-api
- Full-Web App : https://aly-el-badry.github.io/DisasterPrediction/
- Dagshub: https://dagshub.com/Aly-EL-Badry/DisasterPrediction

