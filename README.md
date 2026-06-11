# OmniRecAI 🚀

> End-to-End Recommendation System using Deep Learning Retrieval, FAISS, XGBoost, FastAPI, PostgreSQL, Redis, Docker, MLflow, and React

---

## Overview

OmniRecAI is a production-style recommendation platform that generates personalized recommendations for users using a two-stage recommendation pipeline:

- **Retrieval Stage** – A Deep Learning Two-Tower Model retrieves the most relevant candidate items.
- **Ranking Stage** – An XGBoost Ranker scores and ranks retrieved candidates to generate final recommendations.

The system is served through FastAPI, accelerated with Redis caching, backed by PostgreSQL, tracked using MLflow, containerized with Docker, and visualized through a React dashboard.

---

## Architecture
                 User ID
                    │
                    ▼
         Two-Tower Retrieval Model
               (PyTorch)
                    │
                    ▼
           User Embedding Vector
                    │
                    ▼
              FAISS Search
                    │
                    ▼
         Top Candidate Items
                    │
                    ▼
             XGBoost Ranker
                    │
                    ▼
         Top Recommendations
                    │
                    ▼
              FastAPI API
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
     Redis Cache          PostgreSQL
        │                       │
        └───────────┬───────────┘
                    ▼
              React Dashboard
---

## Key Features

### Deep Learning Retrieval
- Two-Tower Neural Network
- User Embeddings
- Item Embeddings
- Contrastive Learning Style Retrieval
- FAISS Vector Search

### Ranking System
- XGBoost Ranker
- User Features
- Item Features
- Behavioral Features
- Probability-Based Ranking

### Backend
- FastAPI
- REST APIs
- Structured JSON Responses
- Logging
- Metrics Tracking

### Database
- PostgreSQL
- SQLAlchemy ORM
- Repository Pattern

### Caching
- Redis
- Recommendation Caching
- Cache Hit/Miss Monitoring

### MLOps
- MLflow Experiment Tracking
- Model Versioning
- Hyperparameter Logging
- Metric Tracking

### Frontend
- React
- Vite
- Axios
- Recommendation Dashboard

### Deployment
- Docker
- Docker Compose
- Multi-Service Architecture

---

## Dataset

Synthetic large-scale recommendation dataset:

| Metric | Value |
|---|---|
| Users | 10,000 |
| Items | 5,000 |
| Interactions | 500,000 |
| Retrieval Training Pairs | 125,130 |
| Ranking Samples | 500,000 |

**User Features**
- Age
- Premium Membership
- Total Views
- Total Clicks
- Total Purchases

**Item Features**
- Category
- Price
- Rating
- Total Views
- Total Clicks
- Total Purchases

---

## Model Pipeline

### Stage 1: Retrieval

The retrieval model learns user-item similarity using a Two-Tower architecture.

**Input:**
- User Features
- Item Features
- Interaction Labels

**Output:**
- 64-Dimensional User Embedding
- 64-Dimensional Item Embedding

FAISS performs nearest-neighbor search to retrieve relevant candidate items.

### Stage 2: Ranking

The ranker receives retrieved candidates and predicts engagement probability.

**Features:**
- Age
- Premium User
- Price
- Rating
- User Views
- User Clicks
- User Purchases
- Item Views
- Item Clicks
- Item Purchases

**Output:**
- Probability(User Likes Item)

Candidates are sorted by score to generate final recommendations.

---

## Experimental Results

### Retrieval

| Metric | Score |
|---|---|
| Recall@10 | 0.54 |
| Recall@20 | 0.68 |

### Ranking

| Metric | Score |
|---|---|
| AUC | 0.83 |

---

## Technology Stack

| Category | Technologies |
|---|---|
| Machine Learning | PyTorch, XGBoost, NumPy, Pandas, Scikit-Learn |
| Vector Search | FAISS |
| Backend | FastAPI, Uvicorn |
| Database | PostgreSQL, SQLAlchemy |
| Cache | Redis |
| Frontend | React, Vite, Axios |
| MLOps | MLflow |
| Infrastructure | Docker, Docker Compose |

---

## API Endpoints

### Health Check
GET /health
Response:
```json
{
  "status": "healthy",
  "service": "OmniRecAI"
}
```

### Recommendations
GET /recommend/{user_id}
Response:
```json
{
  "user_id": 123,
  "count": 10,
  "recommendations": [...]
}
```

### Metrics
GET /metrics
Response:
```json
{
  "request_count": 10,
  "recommendation_requests": 8,
  "cache_hits": 4,
  "cache_misses": 4
}
```

---

## Running Locally

### Clone Repository

```bash
git clone https://github.com/suryanshbt211/OmniRecAI.git

cd OmniRecAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Backend

```bash
python3 -m uvicorn backend.app:app --reload
```

### Start Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## Running with Docker

Start all services:

```bash
docker compose up --build
```

Services:
FastAPI      → localhost:8000
PostgreSQL   → localhost:5432
Redis        → localhost:6379
React        → localhost:5173
---

## MLflow Tracking

Train ranker with experiment tracking:

```bash
python3 train_ranker_mlflow.py
```

Launch MLflow UI:

```bash
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:
http://127.0.0.1:5000
---

## Project Structure
OmniRecAI
│
├── backend/
│   ├── app.py
│   ├── cache.py
│   ├── database.py
│   ├── repositories.py
│   ├── recommendation_service.py
│   └── models.py
│
├── frontend/
│   ├── src/
│   └── public/
│
├── retrieval/
│   ├── train_retrieval.py
│   ├── recommend.py
│   └── evaluate_retrieval.py
│
├── ranking/
│   ├── train_ranker.py
│   └── build_ranking_dataset.py
│
├── scripts/
│
├── models/
│
├── data/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

---

## Future Improvements

- Hybrid Recommendation System
- Session-Based Recommendations
- Online Learning
- Kubernetes Deployment
- Prometheus Monitoring
- Grafana Dashboards
- A/B Testing Framework
- Real-Time Feature Store
- Feature Registry
- Model Registry

---

## Author

**Suryansh Talukdar**  
MS Computer Science, Purdue University Fort Wayne

**Research Interests:**
- Recommender Systems
- Machine Learning
- Deep Learning
- MLOps
- Responsible AI

**GitHub:** [OmniRecAI Repository](https://github.com/suryanshbt211/OmniRecAI)

---
