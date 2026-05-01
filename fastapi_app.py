import sys
import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

# Create FastAPI app
app = FastAPI(title="Student Performance Predictor", version="1.0")

# Templates
templates = Jinja2Templates(directory="templates")

# Request body model (for API)
class StudentData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float


@app.get('/')
async def home():
    return HTMLResponse(content=open('templates/index.html').read())


@app.get('/predict')
async def predict_form():
    return HTMLResponse(content=open('templates/home.html').read())


@app.post('/predict')
async def predict(
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    try:
        custom_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        pred_df = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Read template and replace placeholder
        html_content = open('templates/home.html').read()
        html_content = html_content.replace(
            '{% if results %}<h2>🎯 Predicted Math Score: {{ results }}</h2>{% endif %}',
            f'<h2>🎯 Predicted Math Score: {float(results[0]):.2f}</h2>'
        )
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        html_content = open('templates/home.html').read()
        html_content = html_content.replace(
            '{% if results %}',
            f'<h2 style="color:red;">Error: {str(e)}</h2>'
        )
        return HTMLResponse(content=html_content, status_code=500)


# API endpoint for JSON requests
@app.post('/api/predict')
async def predict_api(data: StudentData):
    custom_data = CustomData(
        gender=data.gender,
        race_ethnicity=data.race_ethnicity,
        parental_level_of_education=data.parental_level_of_education,
        lunch=data.lunch,
        test_preparation_course=data.test_preparation_course,
        reading_score=data.reading_score,
        writing_score=data.writing_score
    )
    
    pred_df = custom_data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    return {"predicted_math_score": float(results[0])}


@app.get('/health')
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)