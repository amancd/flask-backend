import os
import uvicorn
from fastapi import FastAPI
from models import RecommendationRequest, RecommendationResponse, Assessment
from recommender import recommend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your frontend (adjust the origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-recommender-app.vercel.app"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(payload: RecommendationRequest):
    results = recommend(payload.job_description)
    return {"recommended_assessments": results}

# 👇 Optional for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)