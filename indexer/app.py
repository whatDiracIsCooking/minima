import nltk
import logging
import asyncio
from indexer import Indexer
from pydantic import BaseModel, Field
from typing import List, Optional
from storage import MinimaStore
from async_queue import AsyncQueue
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from fastapi_utilities import repeat_every
from async_loop import index_loop, crawl_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexer = Indexer()
router = APIRouter()
async_queue = AsyncQueue()
MinimaStore.create_db_and_tables()

def init_loader_dependencies():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')

init_loader_dependencies()

class Query(BaseModel):
    query: str


class QueryRequest(BaseModel):
    """Enhanced query request with filtering support."""
    query: str = Field(..., description="Search query text")

    # File type filters
    file_types: Optional[List[str]] = Field(
        None,
        description="Filter by file type: code, doc, data, spreadsheet, presentation"
    )
    file_extensions: Optional[List[str]] = Field(
        None,
        description="Filter by extension: .py, .pdf, .md, etc."
    )

    # Directory filters
    directories: Optional[List[str]] = Field(
        None,
        description="Filter by directory path"
    )

    # Language filters (code files only)
    languages: Optional[List[str]] = Field(
        None,
        description="Filter by language: python, javascript, java, etc."
    )

    # Temporal filters
    modified_after: Optional[str] = Field(
        None,
        description="ISO timestamp - files modified after this date"
    )
    modified_before: Optional[str] = Field(
        None,
        description="ISO timestamp - files modified before this date"
    )

    # Chunk filters
    max_chunk_index: Optional[int] = Field(
        None,
        description="Only return chunks up to this index (0 = first chunk only)"
    )

    # Result limits
    limit: Optional[int] = Field(
        10,
        description="Maximum number of results to return"
    )


@router.post(
    "/query",
    response_description='Query local data storage with optional filtering',
)
async def query(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    try:
        result = indexer.find(
            query=request.query,
            file_types=request.file_types,
            file_extensions=request.file_extensions,
            directories=request.directories,
            languages=request.languages,
            modified_after=request.modified_after,
            modified_before=request.modified_before,
            max_chunk_index=request.max_chunk_index,
            limit=request.limit
        )
        logger.info(f"Found {result.get('total_results', 0)} results for query: {request.query}")
        logger.info(f"Results: {result}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing query: {e}")
        return {"error": str(e)}


@router.post(
    "/embedding", 
    response_description='Get embedding for a query',
)
async def embedding(request: Query):
    logger.info(f"Received embedding request: {request}")
    try:
        result = indexer.embed(request.query)
        logger.info(f"Found {len(result)} results for query: {request.query}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing embedding: {e}")
        return {"error": str(e)}    


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(crawl_loop(async_queue)),
        asyncio.create_task(index_loop(async_queue, indexer))
    ]
    await schedule_reindexing()
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app() -> FastAPI:
    app = FastAPI(
        openapi_url="/indexer/openapi.json",
        docs_url="/indexer/docs",
        lifespan=lifespan
    )
    app.include_router(router)
    return app

async def trigger_re_indexer():
    logger.info("Reindexing triggered")
    try:
        await asyncio.gather(
            crawl_loop(async_queue),
            index_loop(async_queue, indexer)
        )
        logger.info("reindexing finished")
    except Exception as e:
        logger.error(f"error in scheduled reindexing {e}")


@repeat_every(seconds=60*20)
async def schedule_reindexing():
    await trigger_re_indexer()

app = create_app()