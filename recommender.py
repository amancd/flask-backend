import json
from sentence_transformers import SentenceTransformer, util
import torch

# Global caches
_model = None
_doc_embeddings = None
_raw_data = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_data():
    global _raw_data, _doc_embeddings
    if _raw_data is None:
        with open("data/shl_assessments.json", "r") as f:
            _raw_data = json.load(f)["recommended_assessments"]

        documents = [
            f"passage: {item['description']} Skills assessed: {', '.join(item.get('skills', []))}. "
            f"Remote support: {item['remote_support']}. Adaptive: {item['adaptive_support']}. "
            f"Test types: {', '.join(item['test_type'])}. Duration: {item['duration']} minutes."
            for item in _raw_data
        ]
        model = get_model()
        _doc_embeddings = model.encode(documents, convert_to_tensor=True)
    return _raw_data, _doc_embeddings

def recommend(job_desc: str, top_k=10):
    model = get_model()
    raw_data, doc_embeddings = get_data()
    query_embedding = model.encode(f"query: {job_desc}", convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [raw_data[int(i)] for i in top_results]
