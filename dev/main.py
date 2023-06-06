from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn
import os
from sklearn.preprocessing import StandardScaler
import joblib

    
""" Creating the FastAPI Instance. i.e. foundation for our API,
which will be the main part of our project"""

app = FastAPI(title="API")


"""We load a machine learning model and a scaler that help us make predictions based on data."""
model = joblib.load('gbc.pkl',mmap_mode='r')
scaler = joblib.load('scaler.pkl')

"""We define a function that will make predictions using our model and scaler."""
def predict(df, endpoint='simple'):
    # Scaling
    scaled_df = scaler.transform(df)

    # Prediction
    prediction = model.predict_proba(scaled_df)
    highest_proba = prediction.max(axis=1)

    predicted_labels = ["Patient does not have sepsis" if i == 0 else "Patient has Sepsis" for i in highest_proba]
    response = []
    for label, proba in zip(predicted_labels, highest_proba):
        output = {
            "prediction": label,
            "probability of prediction": str(round(proba * 100)) + '%'
        }
        response.append(output)
    return response


"""We create models for the data that our API will work with.
We define what kind of information the data will have.
It's like deciding what information we need to collect and how it should be organized."""


"""These classes define the data models used for API endpoints.
The 'Patient' class represents a single patient's data,
and the 'Patients' class represents a list of patients' data.
The Patients class also includes a class method return_list_of_dict()
that converts the Patients object into a list of dictionaries"""

class Patient(BaseModel):
    Blood_Work_R1: float
    Blood_Pressure: float
    Blood_Work_R3: float
    BMI: float
    Blood_Work_R4: float
    Patient_age: int


"""Next block of code defines different parts of our API and how it responds to different requests.
It sets up a main page with a specific message, provides a checkup endpoint to receive
optional parameters, and sets up prediction endpoints to receive medical data for making predictions,
either for a single patient or multiple patients."""

@app.get("/")
def root():
    return {"API": "This is an API for sepsis prediction."}

# Prediction endpoint (Where we will input our features)
@app.post("/predict")
def predict_sepsis(patient: Patient):

    # Make prediction
    data = pd.DataFrame(patient.dict(), index=[0])
    scaled_data = scaler.transform(data)
    parsed = predict(df=scaled_data)
    return {"output": parsed}


if __name__ == "__main__":
    os.environ["DEBUG"] = "True"  # Enable debug mode
    uvicorn.run("main:app", reload=True)
