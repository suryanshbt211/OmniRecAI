import json
import redis

cache = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)


def get_cached_recommendations(
    user_id
):

    data = cache.get(
        f"user:{user_id}"
    )

    if data:
        return json.loads(data)

    return None


def cache_recommendations(
    user_id,
    recommendations
):

    cache.setex(
        f"user:{user_id}",
        300,
        json.dumps(
            recommendations
        )
    )