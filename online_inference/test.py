import os
from starlette.testclient import TestClient
from app import app


def test_predict(monkeypatch):
    envs = {
        'PATH_TO_MODEL': 'ml_project/models/model.pkl'
    }
    monkeypatch.setattr(os, 'environ', envs)
    request_data = [54,1,3,120,188,0,0,113,0,1.4,1,1,2]
    request_features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    with TestClient(app) as client:
        response = client.post('/predict/', json={"data": [request_data], "features": request_features})
        assert response.status_code == 200
        assert response.json()[0]['сondition'] in (0, 1)
