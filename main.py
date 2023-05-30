from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import logging
import uvicorn
import os

""" Next, we configure a logger, that helps us keep track of what our program is doing.
Like a  notebook where we write down what our program is doing at different points."""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" Creating the FAstAPI Instance. i.e. foundation for our API, 
which will be the main part of our project"""

app = FastAPI(title="API")


"""We load a machine learning model and a scaler.  that help us make predictions based on data."""
def load_model():
    with open("gbc.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

"""We define a function that will make predictions using our model and scaler."""
def predict(df, endpoint="simple"):
    
    return parsed_output

"""We create models for the data that our API will work with. 
We define what kind of information the data will have.
It's like deciding what information we need to collect and how it should be organized."""


"""These classes define the data models used for API endpoints. 
The 'Patient' class represents a single patient's data, 
and the 'Patients' class represents a list of patients' data. 
The Patients class also includes a class method return_list_of_dict() 
that converts the Patients object into a list of dictionaries"""

class Patient(BaseModel):
    Blood_Work_R1: int
    Blood_Pressure: int
    Blood_Work_R3: int
    BMI: float
    Blood_Work_R4: float
    Patient_age: int

class Patients(BaseModel):
    all: list[Patient]

    @classmethod
    def return_list_of_dict(cls, patients: "Patients"):
        return [patient.dict() for patient in patients.all]
