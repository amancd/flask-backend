import uvicorn
from fastapi import FastAPI
from models import RecommendationRequest, RecommendationResponse
from recommender import recommend, load_model_and_data  # ✅ Import this
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Load model/data once during startup
@app.on_event("startup")
def startup_event():
    load_model_and_data()

# ✅ CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-recommender-app.vercel.app"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ Main inference endpoint
@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(payload: RecommendationRequest):
    results = recommend(payload.job_description)
    return {"recommended_assessments": results}
