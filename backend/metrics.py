request_count = 0

recommendation_requests = 0

health_requests = 0

cache_hits = 0

cache_misses = 0


def get_metrics():

    return {
        "request_count": request_count,
        "recommendation_requests": recommendation_requests,
        "health_requests": health_requests,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses
    }