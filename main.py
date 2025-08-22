# main.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import Literal
from model import SimpleNaiveBayes

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('osteoporosis.csv')

class OsteoporosisInput(BaseModel):
    gender: Literal['Male', 'Female']
    hormonal_changes: Literal['Normal', 'Postmenopausal']
    family_history: Literal['Yes', 'No']
    race: Literal['Caucasian', 'Asian', 'African American', 'Hispanic', 'Other']
    body_weight: Literal['Underweight', 'Normal', 'Overweight']
    calcium_intake: Literal['Low', 'Adequate']
    vitamin_d_intake: Literal['Low', 'Adequate']
    physical_activity: Literal['Sedentary', 'Active']
    smoking: Literal['Yes', 'No']
    alcohol_consumption: Literal['Yes', 'No']
    medical_conditions: Literal['None', 'Rheumatoid Arthritis', 'Hyperthyroidism', 'Other']
    medications: Literal['None', 'Corticosteroids', 'Other']
    prior_fractures: Literal['Yes', 'No']

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict')
async def predict(
    request: Request,
    gender: str = Form(...),
    hormonal_changes: str = Form(...),
    family_history: str = Form(...),
    race: str = Form(...),
    body_weight: str = Form(...),
    calcium_intake: str = Form(...),
    vitamin_d_intake: str = Form(...),
    physical_activity: str = Form(...),
    smoking: str = Form(...),
    alcohol_consumption: str = Form(...),
    medical_conditions: str = Form(...),
    medications: str = Form(...),
    prior_fractures: str = Form(...)
):
    # Validate form data using Pydantic
    input_data = OsteoporosisInput(
        gender=gender,
        hormonal_changes=hormonal_changes,
        family_history=family_history,
        race=race,
        body_weight=body_weight,
        calcium_intake=calcium_intake,
        vitamin_d_intake=vitamin_d_intake,
        physical_activity=physical_activity,
        smoking=smoking,
        alcohol_consumption=alcohol_consumption,
        medical_conditions=medical_conditions,
        medications=medications,
        prior_fractures=prior_fractures
    )

    # Convert input data to DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([[
        input_dict['gender'],
        input_dict['hormonal_changes'],
        input_dict['family_history'],
        input_dict['race'],
        input_dict['body_weight'],
        input_dict['calcium_intake'],
        input_dict['vitamin_d_intake'],
        input_dict['physical_activity'],
        input_dict['smoking'],
        input_dict['alcohol_consumption'],
        input_dict['medical_conditions'],
        input_dict['medications'],
        input_dict['prior_fractures']
    ]],
    columns=['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
             'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
             'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
             'Prior Fractures'])

    # Use the model to predict the output
    prediction = model.predict(input_df)
    osteoporosis_result = "Yes" if prediction[0] == 1 else "No"

    return templates.TemplateResponse('index.html', {
        'request': request,
        'prediction_text': f"Osteoporosis Risk: {osteoporosis_result}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)