from loguru import logger

logger.add(
    "logs/api.log",
    rotation="10 MB"
)