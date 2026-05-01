import sys
import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = FastAPI(title="Student Performance Predictor", version="1.0")

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
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Student Performance</title>
    <style>
        body { font-family:Arial; text-align:center; margin-top:100px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height:100vh; }
        .box { background:white; padding:40px; border-radius:15px; display:inline-block; }
        a { display:inline-block; padding:15px 30px; background:#667eea; color:white; text-decoration:none; border-radius:8px; font-size:18px; margin-top:20px; }
    </style>
    </head>
    <body>
        <div class="box">
            <h1>📚 Student Math Score Predictor</h1>
            <p>Predict your math score based on your profile</p>
            <a href="/predict">Start Prediction</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get('/predict')
async def predict_form():
    html_content = open('templates/home.html', encoding='utf-8').read()
    # Remove Jinja2 tags for clean form display
    html_content = html_content.replace(
        '{% if results %}\n<h2>🎯 Predicted Math Score: {{ results }}</h2>\n{% endif %}',
        ''
    )
    return HTMLResponse(content=html_content)


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
        score = float(results[0])
        
        html_content = open('templates/home.html', encoding='utf-8').read()
        html_content = html_content.replace(
            '{% if results %}\n<h2>🎯 Predicted Math Score: {{ results }}</h2>\n{% endif %}',
            f'<h2>🎯 Predicted Math Score: {score:.2f}</h2>'
        )
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        html_content = open('templates/home.html', encoding='utf-8').read()
        html_content = html_content.replace(
            '{% if results %}\n<h2>🎯 Predicted Math Score: {{ results }}</h2>\n{% endif %}',
            f'<h2 style="color:red;">Error: {str(e)}</h2>'
        )
        return HTMLResponse(content=html_content, status_code=500)


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