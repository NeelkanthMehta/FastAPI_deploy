import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote


# Loading Model and creating App object
app = FastAPI()

with open(file='classifier.pkl', mode='rb') as f:
    model = pickle.load(f)

# Index route, opens autometically
@app.get('/')
def index():
    return {'message': 'Hello!'}

# Route with a single parameter, returns the parameter withing a message
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello {name}!'}

# Expose the prediction functionality, make a prediction from
# passing JSON data and return the predicted Bank Note with confidence level
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    kurtosis = data['kurtosis']
    entropy = data['entropy']
    prediction = model.predict([[variance, skewness, kurtosis, entropy]])
    if (prediction[0]>0.5):
        verdict = 'Fake Note'
    else:
        verdict = 'Bank Note'
    return {'class': verdict, 'probability': str(prediction[0])}


if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8000)
