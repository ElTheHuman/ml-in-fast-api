from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load('decision_tree.pkl')

class BankRevenue(BaseModel):
    age:int
    job:str
    marital:str
    education:str
    default:str
    housing:str
    loan:str
    contact:str
    month:str
    day_of_week:str
    duration:float
    campaign:int
    pdays:int
    previous:int
    poutcome:str

@app.get('/')

def root():
    return {'message': 'Welcome to the Bank Revenue Churn Prediction on FastAPI!'}

@app.post('/predict')

def predict(bank:BankRevenue):
    data = bank.dict()
    features = [data['age'], data['job'], data['marital'], data['education'], data['default'], data['housing'], data['loan'], data['contact'], data['month'], data['day_of_week'], data['duration'], data['campaign'], data['pdays'], data['previous'], data['poutcome']]
    transformed_features = model.Transform(features).reshape(1, -1)
    prediction = float(model.Predict(transformed_features)[0])
    return {'prediction': prediction}