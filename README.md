# Experiment 6 – Containerization & API Deployment

## Aim
Package the final ML model in Docker and expose it through a FastAPI prediction service.

## Model
The API serves the tuned Random Forest Regressor selected in Experiment 4 for Tomato retail-price prediction.

## Files
- `main.py` – FastAPI application with `/predict`
- `best_model.pkl` – trained model artifact
- `requirements.txt` – Python dependencies
- `Dockerfile` – container image definition
- `sample_input.json` – sample POST payload
- `test_api.py` – local API test script
- `Experiment_6_Containerization_API_Deployment.ipynb` – complete experiment notebook

## Run locally without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:
`http://127.0.0.1:8000/docs`

Or run:
```bash
python test_api.py
```

## Build and run Docker image

```bash
docker build -t food-price-api .
docker run -d --name food-price-api-container -p 8000:8000 food-price-api
python test_api.py
```

Stop and remove:
```bash
docker stop food-price-api-container
docker rm food-price-api-container
```
