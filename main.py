from fastapi import FastAPI
import uvicorn
import pickle

app = FastAPI()



@app.get("/")
def read_root():
    return {"Hello": "World"}




with open("C:/Users/GilB/OneDrive/Documents/Git Repo/Machine-Learning-API-using-FastAPI/Key_components/gbc.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict(data: dict):
    # Assuming your model expects input features as a dictionary
    features = data["features"]
    
    # Perform prediction using your machine learning model
    prediction = model.predict([features])
    
    return {"prediction": prediction}

