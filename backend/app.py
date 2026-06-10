from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.logger import logger
from backend.recommendation_service import (
    get_recommendations
)

from backend import metrics

app = FastAPI(
    title="OmniRecAI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():

    return {
        "message": "Welcome to OmniRecAI",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    metrics.health_requests += 1

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy",
        "service": "OmniRecAI"
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int
):

    metrics.request_count += 1
    metrics.recommendation_requests += 1

    logger.info(
        f"Recommendation requested for user_id={user_id}"
    )

    recommendations = (
        get_recommendations(
            user_id
        )
    )

    return {
        "user_id": user_id,
        "count": len(
            recommendations
        ),
        "recommendations": recommendations
    }


@app.get("/metrics")
def metrics_api():

    logger.info(
        "Metrics endpoint requested"
    )

    return metrics.get_metrics()