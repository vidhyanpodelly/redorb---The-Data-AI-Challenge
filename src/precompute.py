
import json
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = ROOT_DIR / 'data' / 'candidates.jsonl'
EMBEDDINGS_PATH = ROOT_DIR / 'models' / 'candidate_embeddings.npy'
IDS_PATH = ROOT_DIR / 'models' / 'candidate_ids.json'
MODEL_NAME = 'BAAI/bge-small-en-v1.5'

def main():
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Loading candidates and preparing text...")
    texts = []
    ids = []
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=100000):
            c = json.loads(line)
            ids.append(c['candidate_id'])
            
            # Focused text for embedding
            skills_str = ", ".join([s['name'] for s in c['skills']])
            career_str = " ".join([j['description'] for j in c['career_history']])
            # Prioritize headline and summary
            text = f"Candidate: {c['profile']['headline']}. Summary: {c['profile']['summary']}. Skills: {skills_str}. Career: {career_str}"
            texts.append(text)
            
    print(f"Encoding {len(texts)} candidates using multi-process pool...")
    # Start multi-process pool
    pool = model.start_multi_process_pool()
    
    # Encode using multi-process
    embeddings = model.encode_multi_process(texts, pool, batch_size=128)
    
    # Stop multi-process pool
    model.stop_multi_process_pool(pool)
    
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    print(f"Saving IDs to {IDS_PATH}...")
    with open(IDS_PATH, 'w') as f:
        json.dump(ids, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
