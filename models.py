from pydantic import BaseModel
from typing import List

class RecommendationRequest(BaseModel):
    job_description: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]
