
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = ROOT_DIR / 'data' / 'candidates.jsonl'
EMBEDDINGS_PATH = ROOT_DIR / 'models' / 'candidate_embeddings.npy'
IDS_PATH = ROOT_DIR / 'models' / 'candidate_ids.json'
OUTPUT_PATH = ROOT_DIR / 'outputs' / 'submission.csv'
MODEL_NAME = 'BAAI/bge-small-en-v1.5'

# JD Summary for Semantic Search
JD_CONTENT = """
Senior AI Engineer Founding Team. 
Expertise in modern ML systems: embeddings, retrieval, ranking, search systems, vector databases, hybrid retrieval. 
Production experience: shipping production systems, handling embedding drift, index refresh, retrieval-quality regression. 
Tools: Python, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. 
Evaluation: NDCG, MRR, MAP, offline-to-online correlation, A/B testing. 
Startup mindset: shipper, fast execution, product thinking, building from scratch.
"""

# Keywords for heuristic scoring
CORE_AI = ['embedding', 'retrieval', 'ranking', 'vector search', 'hybrid search', 'rag', 'llm', 'rerank']
PROD = ['production', 'deployed', 'scaling', 'monitoring', 'latency', 'pipeline', 'infrastructure']
EVAL = ['ndcg', 'map', 'mrr', 'evaluation', 'a/b testing', 'offline evaluation']
STARTUP = ['startup', 'founding', 'early stage', 'ownership', 'shipped', 'fast execution']

def get_risk_score(c):
    risk = 0.0
    yoe = c['profile']['years_of_experience']
    
    # 1. Experience vs Career Duration
    total_months = sum(job['duration_months'] for job in c['career_history'])
    actual_years = total_months / 12.0
    if actual_years > yoe + 2.0: risk += 0.5
    
    # 2. Skill duration vs Experience
    for skill in c['skills']:
        if skill['duration_months'] / 12.0 > yoe + 1.0:
            risk += 0.4
            break
            
    # 3. Impossible overlaps
    jobs = sorted(c['career_history'], key=lambda x: x['start_date'])
    for i in range(len(jobs) - 1):
        if not jobs[i]['is_current'] and jobs[i]['end_date'] and jobs[i+1]['start_date']:
            if jobs[i]['end_date'] > jobs[i+1]['start_date']:
                d1 = datetime.strptime(jobs[i]['end_date'], '%Y-%m-%d')
                d2 = datetime.strptime(jobs[i+1]['start_date'], '%Y-%m-%d')
                if (d1 - d2).days > 60:
                    risk += 0.3
                    
    # 4. Keyword stuffers
    if len(c['skills']) > 50: risk += 0.2
    
    return min(risk, 1.0)

def get_heuristic_scores(c):
    skills_str = " ".join([s['name'] for s in c['skills']]).lower()
    career_str = " ".join([j['description'] for j in c['career_history']]).lower()
    all_text = f"{c['profile']['headline']} {c['profile']['summary']} {skills_str} {career_str}".lower()
    
    # Use unique keyword matching to prevent keyword stuffing reward
    s_ai = sum(1 for kw in CORE_AI if kw in all_text) / len(CORE_AI)
    s_prod = sum(1 for kw in PROD if kw in career_str) / len(PROD)
    s_eval = sum(1 for kw in EVAL if kw in all_text) / len(EVAL)
    s_startup = sum(1 for kw in STARTUP if kw in all_text) / len(STARTUP)
    
    # Negative signals: pure academic without production
    neg_score = 0
    if any(kw in all_text for kw in ['academic', 'postdoc', 'professor', 'research lab']):
        if not any(kw in career_str for kw in ['production', 'shipped', 'deployed']):
            neg_score += 0.2
            
    # Service company penalty
    service_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    if any(kw in c['profile']['current_company'].lower() for kw in service_companies):
        neg_score += 0.1
        
    return s_ai, s_prod, s_eval, s_startup, neg_score

def get_behavior_score(s):
    score = (s['recruiter_response_rate'] * 0.4 + 
             s['profile_completeness_score'] / 100.0 * 0.2 + 
             s['interview_completion_rate'] * 0.2)
    
    if s['open_to_work_flag']: score += 0.1
    if s['github_activity_score'] > 50: score += 0.1
    
    return score

def generate_reasoning(c, rank):
    yoe = c['profile']['years_of_experience']
    title = c['profile']['current_title']
    top_skills = [s['name'] for s in c['skills'][:3]]
    
    # Variety of templates
    templates = [
        f"Senior professional with {yoe}y experience as {title}, showing deep expertise in {', '.join(top_skills)}.",
        f"Expert in {', '.join(top_skills)} with a strong {yoe}-year track record as {title}.",
        f"Currently {title}, this candidate brings {yoe} years of expertise specifically in {top_skills[0]} and {top_skills[1]}.",
        f"High-impact {title} with {yoe} years in the field. Key skills include {', '.join(top_skills)}."
    ]
    
    # Pick template based on candidate_id hash for consistency but variety
    import hashlib
    h = int(hashlib.md5(c['candidate_id'].encode()).hexdigest(), 16)
    reason = templates[h % len(templates)]
    
    history = " ".join([j['description'] for j in c['career_history']]).lower()
    
    # Add specific factual observations
    observations = []
    if any(kw in history for kw in ['production', 'deploy', 'scale', 'pipeline']):
        observations.append("Strong evidence of shipping production ML systems.")
    
    if any(kw in history for kw in ['ndcg', 'map', 'mrr', 'a/b test']):
        observations.append("Hands-on experience with ranking evaluation metrics.")
        
    if c['redrob_signals']['recruiter_response_rate'] > 0.8:
        observations.append("Highly responsive on the Redrob platform.")
        
    if observations:
        # Pick one observation to add
        reason += " " + observations[h % len(observations)]
        
    return reason.strip()[:250]

def main():
    print("Loading data...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print("Embeddings not found! Run precompute.py first.")
        return
        
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(IDS_PATH, 'r') as f:
        ids = json.load(f)
    
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    
    print("Encoding JD...")
    model = SentenceTransformer(MODEL_NAME)
    jd_emb = model.encode([JD_CONTENT])[0]
    
    print("Computing semantic similarities...")
    similarities = cosine_similarity([jd_emb], embeddings)[0]
    
    print("Loading candidates and scoring...")
    candidates = []
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= len(ids): break # Use only what we precomputed
            candidates.append(json.loads(line))
            
    all_results = []
    for i, c in enumerate(tqdm(candidates)):
        cid = c['candidate_id']
        semantic_score = similarities[id_to_idx[cid]]
        
        s_ai, s_prod, s_eval, s_startup, neg = get_heuristic_scores(c)
        behavior = get_behavior_score(c['redrob_signals'])
        risk = get_risk_score(c)
        
        # Weighted Fusion
        # Semantic(0.4) + AI(0.2) + Prod(0.2) + Eval(0.1) + Startup(0.1)
        composite_score = (semantic_score * 0.4 + 
                           s_ai * 0.2 + 
                           s_prod * 0.2 + 
                           s_eval * 0.1 + 
                           s_startup * 0.1 - 
                           neg)
        
        # Adjust by behavior and risk
        final_score = composite_score * (0.8 + 0.2 * behavior) * (1.0 - risk)
        
        all_results.append({
            'candidate_id': cid,
            'score': final_score,
            'candidate': c
        })
        
    # Sort by score descending, then candidate_id ascending for ties
    all_results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    top_100 = all_results[:100]
    
    output = []
    for i, res in enumerate(top_100):
        output.append({
            'candidate_id': res['candidate_id'],
            'rank': i + 1,
            'score': round(float(res['score']), 6),
            'reasoning': generate_reasoning(res['candidate'], i+1)
        })
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(output)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved to {OUTPUT_PATH}")
    
    # Export to Excel
    try:
        excel_path = OUTPUT_PATH.with_suffix('.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Submission (Excel) saved to {excel_path}")
    except Exception as e:
        print(f"Could not save Excel version: {e}")

if __name__ == "__main__":
    main()
