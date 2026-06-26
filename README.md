# Redrob Intelligent Candidate Discovery & Ranking System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](requirements.txt)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, high-fidelity candidate discovery and ranking system designed for the **Redrob AI Founding Team Senior AI Engineer** role. This system employs a multi-stage hybrid ranking strategy that fuses semantic retrieval with custom heuristics, behavioral metrics, and a robust honeypot/risk detector to produce validated, high-impact rankings.

---

## 📖 Challenge Description & Project Overview

Finding the right talent in artificial intelligence is frequently subverted by over-optimized profiles, "AI tourists" utilizing generic boilerplate, keyword stuffers, and inconsistent career timelines ("honeypot profiles"). 

This project solves these issues by implementing a **6-component hybrid ranking pipeline**. Instead of relying solely on raw vector similarity, the engine scores technical competency, logs proof of shipping real production systems, accounts for platform behavior signals, and applies strict heuristic penalization rules to identify top-tier engineers.

---

## 🛠️ System Architecture

The pipeline consists of three sequential stages: **Precomputation**, **Ranking & Fusion**, and **Validation**.

```
                       +-----------------------------+
                       |   candidates.jsonl (Raw)    |
                       +--------------+--------------+
                                      |
                           [Step 1: Precompute]
                                      v
                       +--------------+--------------+
                       |    BGE-Small Embeddings     |
                       +--------------+--------------+
                                      |
                            [Step 2: rank.py] <--------- job_description.docx
                                      v
            +-------------------------+-------------------------+
            |                                                   |
    [Semantic Similarity (40%)]                     [Heuristics Matching (60%)]
            |                                       - Tech Domain Match (20%)
            |                                       - Production Ready (20%)
            |                                       - Evaluation Expertise (10%)
            |                                       - Shipper Mindset (10%)
            |                                                   |
            +-------------------------+-------------------------+
                                      |
                                      v
                         +------------+------------+
                         |      Composite Score    |
                         +------------+------------+
                                      |
                            [Behavior Multiplier] (0.8 - 1.0)
                                      |
                                      v
                            [Risk Penalty Filter] (1.0 - Risk)
                                      |
                                      v
                         +------------+------------+
                         |       Final Score       |
                         +------------+------------+
                                      |
                                      v
                             [Top 100 Reranking]
                                      |
                            [Step 3: Validation]
                                      v
                         +------------+------------+
                         |   submission.csv/xlsx   |
                         +-------------------------+
```

*For higher resolution visual graphics, inspect the generated documentation assets:*
- **System Architecture**: [docs/Architecture.png](docs/Architecture.png)
- **Execution Workflow**: [docs/Workflow.png](docs/Workflow.png)
- **Ranking Pipeline math**: [docs/RankingPipeline.png](docs/RankingPipeline.png)

---

## 📁 Repository Directory Structure

```
AI-Candidate-Discovery/
│
├── README.md                           # Main project documentation and setup
├── LICENSE                             # MIT Open Source License
├── requirements.txt                    # Project python library dependencies
├── .gitignore                          # Rules to keep dataset and binaries out of git
│
├── docs/                               # Documentation assets and diagrams
│   ├── Architecture.png                # Structural block diagram of the system
│   ├── Workflow.png                    # E2E process run-flow diagram
│   ├── RankingPipeline.png             # Math scoring & fusion visualization
│   ├── README.docx                     # Reference documentation in docx format
│   ├── job_description.docx            # Original job requirements
│   ├── submission_spec.docx            # Format and constraint specifications
│   ├── redrob_signals_doc.docx         # Behavior signals documentation
│   └── Idea Submission Template _ Redrob.pptx # Official PowerPoint template
│
├── src/                                # Source Python files
│   ├── extract_docs.py                 # Extractor helper for docx files
│   ├── precompute.py                   # BGE-small embedding generation
│   ├── rank.py                         # Core ranking and scoring engine
│   ├── final_run.py                    # Orchestrator running precompute -> rank -> validate
│   └── validate_submission.py          # Strict checker for competition compliance
│
├── data/                               # Dataset directory (Add dataset here)
│   ├── candidate_schema.json           # JSON Schema defining candidates attributes
│   ├── sample_candidates.json          # Small dataset for local evaluation
│   ├── sample_submission.csv           # Formatting reference target
│   └── candidates.jsonl                # [EXCLUDED FROM GIT] Main challenge dataset (487MB)
│
├── models/                             # Output model checkpoints & embeddings (Git ignored)
│   ├── candidate_embeddings.npy        # Raw vector representations (153MB)
│   └── candidate_ids.json              # Mapping keys for vectorized rows
│
└── outputs/                            # Target output directory (Git ignored)
    ├── submission.csv                  # final CSV for challenge upload
    └── submission.xlsx                 # final Excel formatting version
```

---

## ⚙️ Installation & Requirements

### Software Environment
- Python 3.11+
- Operating System: Windows, macOS, or Linux
- Disk space: ~1GB (required for raw `.jsonl` and generated `.npy` embeddings)

### Dependencies
Dependencies are listed in [requirements.txt](requirements.txt). To install:
```bash
pip install -r requirements.txt
```

*Note: For the Excel export feature, `openpyxl` is recommended.*

---

## 🚀 How to Run the Pipeline

Follow these steps to run the pipeline end-to-end:

### 1. Place the Dataset
Download `candidates.jsonl` (487MB) from the portal and place it in the `data/` folder:
```bash
# Correct path structure
data/candidates.jsonl
```

### 2. Precompute Candidate Embeddings
The semantic retrieval module uses a multi-process pool to encode candidate resumes. Run:
```bash
python src/precompute.py
```
This generates:
- `models/candidate_embeddings.npy` (Candidate vector embeddings index)
- `models/candidate_ids.json` (Array matching row indexes to candidate IDs)

### 3. Run Ranking Engine
Compute scores, apply heuristics, integrate behavioral modifiers, and run the risk detector:
```bash
python src/rank.py
```
This writes:
- `outputs/submission.csv`
- `outputs/submission.xlsx`

### 4. Run Submission Validator
Ensure strict compliance with challenge guidelines:
```bash
python src/validate_submission.py outputs/submission.csv
```

### 5. Consolidated Orchestrated Run
You can run the full execution workflow automatically (waiting for precomputations, running ranking, and validating) by executing:
```bash
python src/final_run.py
```

---

## 🧠 Detailed Pipeline Mechanics

The candidate ranking score ($S$) is computed using a **Weighted Fusion** ($C$) modified by a **Behavioral Multiplier** ($B$) and a **Risk Coefficient** ($R$):

$$S = C \times B \times (1.0 - R)$$

### 1. Weighted Composite Score ($C$)
The composite score is compiled from five sub-heuristics:
- **Semantic Fit (40%)**: Cosine similarity between candidate profiles and the parsed Job Description using `BAAI/bge-small-en-v1.5`.
- **Technical AI Domain Match (20%)**: Keyword/phrase scanning for search, retrieval, indexing, LLM, and ranking frameworks.
- **Production Readiness (20%)**: Search for evidence of deploying, hosting, scaling, and maintaining ML pipelines in production.
- **Evaluation Expertise (10%)**: Checks for explicit understanding of metrics like NDCG, MAP, MRR, and offline/online A/B testing.
- **Shipper/Startup Mindset (10%)**: Rewards small-team ownership, fast execution history, and startup backgrounds.

### 2. Behavioral Multiplier ($B$)
Calibrates platform-specific responsiveness, capping candidates who look good on paper but do not engage:
- Multiplier Range: $[0.8, 1.0]$ based on response rate, profile completeness, interview completions, and "Open to Work" status.

### 3. Risk Coefficient ($R$)
Detects and penalizes:
- **Honeypot Timelines**: Impossible timeline overlaps, or having skill durations exceeding total experience.
- **Keyword Stuffers**: Profiles claiming over 50 skills with minimal supporting details in work histories.
- **Service-Company Bias**: Applied to current/past corporate staffing agencies to skew ranking towards high-ownership startup profiles.

---

## 📊 Reproducibility & Output Format

### Output Specifications
The final `outputs/submission.csv` contains exactly **100 rows** after the header with columns:
- `candidate_id`: Stripped ID of format `CAND_XXXXXXX`
- `rank`: Integer from 1 to 100
- `score`: Float rounded to 6 decimal places (strictly non-increasing)
- `reasoning`: Descriptive, non-hallucinated justification explaining *why* the candidate was ranked there.

### Verification Compliance
Our scoring guarantees:
- Non-increasing scores across ranks.
- Candidate ID alphabetical sorting as a tie-breaker for identical scores.
- Zero duplicate candidate IDs or ranks.

---

## ⏱️ Runtime & Explainability

- **Precomputation**: Encoding 100K profiles takes ~15 minutes on a standard 12-core CPU.
- **Ranking Runtime**: ~8 seconds (fits easily within the 5-minute competition limits).
- **Explainability**: Custom md5-hash variations ensure unique, factual candidate descriptions containing actual candidate details, experience years, and platform signals, providing recruiters with reliable feedback.

---

## 🔮 Future Improvements

1. **Reranker Integration**: Fusing BGE-small embeddings with a Cross-Encoder reranking step (e.g., `BAAI/bge-reranker-base`) for top 500 candidates.
2. **Dynamic JD Parsing**: Automatically extracting core keywords from docx files instead of using constants.
3. **Fuzzy Name Matching**: Standardizing corporate profiles to improve service-company detection.

---

## 🛠️ Development Process

This project was developed through iterative engineering, experimentation, and manual implementation. AI-assisted tools were used selectively for brainstorming, documentation refinement, and productivity, while the system design, implementation, validation, testing, and technical decisions were performed and verified by the project author.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
