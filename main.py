from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = FastAPI()

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict_news(news: NewsInput):
    vectorized = vectorizer.transform([news.text])
    prediction = model.predict(vectorized)
    return {"prediction": "Real" if prediction[0] == 1 else "Fake"}
