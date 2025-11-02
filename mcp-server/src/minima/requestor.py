import httpx
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow override via env var for Docker container usage
INDEXER_HOST = os.getenv("MINIMA_INDEXER_HOST", "localhost")
REQUEST_DATA_URL = f"http://{INDEXER_HOST}:8001/query"
REQUEST_HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

async def request_data(query):
    payload = {
        "query": query
    }
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Requesting data from indexer with query: {query}")
            response = await client.post(REQUEST_DATA_URL, 
                                         headers=REQUEST_HEADERS, 
                                         json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Received data: {data}")
            return data

        except Exception as e:
            logger.error(f"HTTP error: {e}")
            return { "error": str(e) }