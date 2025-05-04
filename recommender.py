import json
from sentence_transformers import SentenceTransformer, util

# Load data
with open("data/shl_assessments.json", "r") as f:
    raw_data = json.load(f)["recommended_assessments"]

# Load E5 model
model = SentenceTransformer("intfloat/e5-small-v2")

# Prepare enhanced documents
documents = [
    f"passage: {item['description']} Skills assessed: {', '.join(item.get('skills', []))}. "
    f"Remote support: {item['remote_support']}. Adaptive: {item['adaptive_support']}. "
    f"Test types: {', '.join(item['test_type'])}. Duration: {item['duration']} minutes."
    for item in raw_data
]

# Precompute document embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=True)

def recommend(job_desc: str, top_k=10):
    query_embedding = model.encode(f"query: {job_desc}", convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [raw_data[int(i)] for i in top_results]
