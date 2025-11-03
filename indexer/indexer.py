import os
import uuid
import torch
import logging
import time
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, Range
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    UnstructuredPowerPointLoader,
)

from storage import MinimaStore, IndexingStatus

logger = logging.getLogger(__name__)


@dataclass
class Config:
    EXTENSIONS_TO_LOADERS = {
        ".pdf": PyMuPDFLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
        # Code files (use TextLoader)
        ".py": TextLoader,
        ".js": TextLoader,
        ".java": TextLoader,
        ".cpp": TextLoader,
        ".c": TextLoader,
        ".h": TextLoader,
        ".hpp": TextLoader,
        ".rs": TextLoader,
    }
    
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    
    START_INDEXING = os.environ.get("START_INDEXING")
    LOCAL_FILES_PATH = os.environ.get("LOCAL_FILES_PATH")
    CONTAINER_PATH = os.environ.get("CONTAINER_PATH")
    QDRANT_COLLECTION = "mnm_storage"
    QDRANT_BOOTSTRAP = "qdrant"
    EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID")
    EMBEDDING_SIZE = os.environ.get("EMBEDDING_SIZE")
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200

class Indexer:
    def __init__(self):
        self.config = Config()
        self.qdrant = self._initialize_qdrant()
        self.embed_model = self._initialize_embeddings()
        self.document_store = self._setup_collection()
        self.text_splitter = self._initialize_text_splitter()

    def _initialize_qdrant(self) -> QdrantClient:
        return QdrantClient(host=self.config.QDRANT_BOOTSTRAP)

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_ID,
            model_kwargs={'device': self.config.DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    def _setup_collection(self) -> QdrantVectorStore:
        if not self.qdrant.collection_exists(self.config.QDRANT_COLLECTION):
            self.qdrant.create_collection(
                collection_name=self.config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.config.EMBEDDING_SIZE,
                    distance=Distance.COSINE
                ),
            )

        # Create payload indexes for filtering
        payload_indexes = [
            "fpath",
            "metadata.file_ext",
            "metadata.file_name",
            "metadata.file_type",      # NEW - Phase 2
            "metadata.directory",      # NEW - Phase 2
            "metadata.language",       # NEW - Phase 2
        ]

        for field_name in payload_indexes:
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.config.QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema="keyword"
                )
                logger.info(f"Created payload index for {field_name}")
            except Exception as e:
                logger.info(f"Payload index for {field_name} may already exist: {e}")

        # Numeric indexes for range queries
        numeric_indexes = [
            "metadata.chunk_index",
            "metadata.file_size",           # NEW - Phase 2
            "metadata.modified_timestamp",  # NEW - Phase 2 (Unix timestamp)
        ]

        for field_name in numeric_indexes:
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.config.QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema="integer"
                )
                logger.info(f"Created payload index for {field_name}")
            except Exception as e:
                logger.info(f"Payload index for {field_name} may already exist: {e}")

        return QdrantVectorStore(
            client=self.qdrant,
            collection_name=self.config.QDRANT_COLLECTION,
            embedding=self.embed_model,
        )

    def _create_loader(self, file_path: str):
        file_extension = Path(file_path).suffix.lower()
        loader_class = self.config.EXTENSIONS_TO_LOADERS.get(file_extension)

        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return loader_class(file_path=file_path)

    def _classify_file_type(self, ext: str) -> str:
        """
        Classify file into broad categories for filtering.

        Args:
            ext: File extension (e.g., '.py', '.pdf')

        Returns:
            Category: 'code', 'doc', 'data', 'spreadsheet', 'presentation', 'other'
        """
        type_map = {
            'code': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                     '.h', '.hpp', '.rs', '.go', '.rb', '.php', '.swift', '.kt'],
            'doc': ['.md', '.txt', '.pdf', '.docx', '.doc', '.rtf'],
            'data': ['.csv', '.json', '.xml', '.yaml', '.yml', '.toml'],
            'spreadsheet': ['.xlsx', '.xls', '.ods'],
            'presentation': ['.pptx', '.ppt', '.odp'],
        }

        for file_type, extensions in type_map.items():
            if ext.lower() in extensions:
                return file_type

        return 'other'

    def _detect_language(self, ext: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            ext: File extension

        Returns:
            Language name or 'unknown'
        """
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
        }

        return language_map.get(ext.lower(), 'unknown')

    def _process_file(self, loader) -> List[str]:
        try:
            documents = loader.load_and_split(self.text_splitter)
            if not documents:
                logger.warning(f"No documents loaded from {loader.file_path}")
                return []

            file_path = loader.file_path
            file_ext = Path(file_path).suffix.lower()
            total_chunks = len(documents)

            # Phase 2: Add classification metadata
            file_type = self._classify_file_type(file_ext)
            file_size = os.path.getsize(file_path)
            modified_timestamp = int(os.path.getmtime(file_path))  # Unix timestamp for range queries
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()  # ISO string for display
            directory = os.path.dirname(file_path)

            # Add enhanced metadata to each chunk
            for idx, doc in enumerate(documents):
                metadata = {
                    # Phase 1 fields
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_ext': file_ext,
                    'chunk_index': idx,
                    'total_chunks': total_chunks,

                    # Phase 2 fields
                    'file_type': file_type,
                    'file_size': file_size,
                    'modified_time': modified_time,  # ISO string for display
                    'modified_timestamp': modified_timestamp,  # Unix timestamp for filtering
                    'directory': directory,
                }

                # Add language for code files
                if file_type == 'code':
                    metadata['language'] = self._detect_language(file_ext)

                doc.metadata.update(metadata)

            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            ids = self.document_store.add_documents(documents=documents, ids=uuids)

            logger.info(f"Successfully processed {len(ids)} documents from {loader.file_path}")
            return ids

        except Exception as e:
            logger.error(f"Error processing file {loader.file_path}: {str(e)}")
            return []

    def index(self, message: Dict[str, any]) -> None:
        start = time.time()
        path, file_id, last_updated_seconds = message["path"], message["file_id"], message["last_updated_seconds"]
        logger.info(f"Processing file: {path} (ID: {file_id})")
        indexing_status: IndexingStatus = MinimaStore.check_needs_indexing(fpath=path, last_updated_seconds=last_updated_seconds)
        if indexing_status != IndexingStatus.no_need_reindexing:
            logger.info(f"Indexing needed for {path} with status: {indexing_status}")
            try:
                if indexing_status == IndexingStatus.need_reindexing:
                    logger.info(f"Removing {path} from index storage for reindexing")
                    self.remove_from_storage(files_to_remove=[path])
                loader = self._create_loader(path)
                ids = self._process_file(loader)
                if ids:
                    logger.info(f"Successfully indexed {path} with IDs: {ids}")
            except Exception as e:
                logger.error(f"Failed to index file {path}: {str(e)}")
        else:
            logger.info(f"Skipping {path}, no indexing required. timestamp didn't change")
        end = time.time()
        logger.info(f"Processing took {end - start} seconds for file {path}")

    def purge(self, message: Dict[str, any]) -> None:
        existing_file_paths: list[str] = message["existing_file_paths"]
        files_to_remove = MinimaStore.find_removed_files(existing_file_paths=set(existing_file_paths))
        if len(files_to_remove) > 0:
            logger.info(f"purge processing removing old files {files_to_remove}")
            self.remove_from_storage(files_to_remove)
        else:
            logger.info("Nothing to purge")

    def remove_from_storage(self, files_to_remove: list[str]):
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="fpath",
                    match=MatchValue(value=fpath)
                )
                for fpath in files_to_remove
            ]
        )
        response = self.qdrant.delete(
            collection_name=self.config.QDRANT_COLLECTION,
            points_selector=filter_conditions,
            wait=True
        )
        logger.info(f"Delete response for {len(files_to_remove)} for files: {files_to_remove} is: {response}")

    def find(
        self,
        query: str,
        file_types: List[str] = None,
        file_extensions: List[str] = None,
        directories: List[str] = None,
        languages: List[str] = None,
        modified_after: str = None,
        modified_before: str = None,
        max_chunk_index: int = None,
        limit: int = 10
    ) -> Dict[str, any]:
        """
        Enhanced search with metadata filtering.

        Args:
            query: Search query string
            file_types: Filter by file type (code, doc, data, etc.)
            file_extensions: Filter by extension (.py, .pdf, etc.)
            directories: Filter by directory path
            languages: Filter by programming language
            modified_after: ISO timestamp - files modified after this date
            modified_before: ISO timestamp - files modified before this date
            max_chunk_index: Only return first N chunks per file
            limit: Maximum results to return

        Returns:
            Dict with links, output, and metadata
        """
        try:
            logger.info(f"Searching for: {query} with filters")

            # Build filter conditions
            must_conditions = []

            if file_types:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.file_type",
                        match=MatchAny(any=file_types)
                    )
                )

            if file_extensions:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.file_ext",
                        match=MatchAny(any=file_extensions)
                    )
                )

            if directories:
                # Match any of the provided directories
                dir_conditions = [
                    FieldCondition(
                        key="metadata.directory",
                        match=MatchValue(value=dir_path)
                    )
                    for dir_path in directories
                ]
                if len(dir_conditions) > 1:
                    must_conditions.append(
                        Filter(should=dir_conditions, min_should_match=1)
                    )
                else:
                    must_conditions.extend(dir_conditions)

            if languages:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.language",
                        match=MatchAny(any=languages)
                    )
                )

            if modified_after:
                # Convert ISO timestamp to Unix timestamp for filtering
                try:
                    modified_after_ts = int(datetime.fromisoformat(modified_after.replace('Z', '+00:00')).timestamp())
                    must_conditions.append(
                        FieldCondition(
                            key="metadata.modified_timestamp",
                            range=Range(gte=modified_after_ts)
                        )
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid modified_after timestamp: {modified_after}, error: {e}")

            if modified_before:
                # Convert ISO timestamp to Unix timestamp for filtering
                try:
                    modified_before_ts = int(datetime.fromisoformat(modified_before.replace('Z', '+00:00')).timestamp())
                    must_conditions.append(
                        FieldCondition(
                            key="metadata.modified_timestamp",
                            range=Range(lte=modified_before_ts)
                        )
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid modified_before timestamp: {modified_before}, error: {e}")

            if max_chunk_index is not None:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.chunk_index",
                        range=Range(lte=max_chunk_index)
                    )
                )

            # Create filter object
            filter_obj = Filter(must=must_conditions) if must_conditions else None

            # Search with filters
            found = self.document_store.search(
                query,
                search_type="similarity",
                filter=filter_obj,
                k=limit
            )

            if not found:
                logger.info("No results found")
                return {"links": [], "output": "", "metadata": []}

            links = set()
            results = []
            metadata_info = []

            for item in found:
                meta = item.metadata
                path = meta["file_path"].replace(
                    self.config.CONTAINER_PATH,
                    self.config.LOCAL_FILES_PATH
                )

                # Collect file information
                file_info = {
                    "file": path,
                    "file_name": meta.get("file_name"),
                    "file_type": meta.get("file_type"),
                    "file_ext": meta.get("file_ext"),
                    "chunk": f"{meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', '?')}",
                    "modified": meta.get("modified_time"),
                    "language": meta.get("language", "N/A"),
                }

                links.add(f"file://{path}")
                results.append(item.page_content)
                metadata_info.append(file_info)

            output = {
                "links": list(links),
                "output": ". ".join(results),
                "metadata": metadata_info,
                "total_results": len(found),
                "filters_applied": {
                    "file_types": file_types,
                    "file_extensions": file_extensions,
                    "directories": directories,
                    "languages": languages,
                    "max_chunk_index": max_chunk_index,
                }
            }

            logger.info(f"Found {len(found)} results with filters")
            return output

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": "Unable to find anything for the given query"}

    def embed(self, query: str):
        return self.embed_model.embed_query(query)