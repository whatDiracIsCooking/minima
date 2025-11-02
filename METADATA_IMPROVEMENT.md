# Metadata Optimization Analysis - Minima + Qdrant RAG System

## Executive Summary

Based on codebase analysis, the current Minima indexer **significantly underutilizes** Qdrant's metadata filtering capabilities. While the infrastructure exists (proven by the deletion logic), metadata is not used for query filtering, and only minimal metadata is stored.

## Current State: Underutilized Metadata

### What's Actually Happening:

**1. Metadata Construction is Minimal**
```python
# minima/indexer/indexer.py:124-142
for doc in documents:
    doc.metadata['file_path'] = loader.file_path  # That's it!
```
- **Only one field added**: `file_path`
- No chunk position, file type, timestamps, page numbers, etc.

**2. Qdrant Filtering is NOT Used for Queries**
```python
# minima/indexer/indexer.py:192-222
def find(self, query: str):
    # Pure similarity search - NO metadata filtering!
    found = self.document_store.search(query, search_type="similarity")
```
- Metadata is only used **after retrieval** to generate file links
- No pre-filtering by file type, path, date, etc.

**3. BUT... Filtering DOES Work (for deletion)**
```python
# minima/indexer/indexer.py:175-190
filter_conditions = Filter(
    must=[FieldCondition(key="fpath", match=MatchValue(value=fpath))]
)
self.qdrant.delete(..., points_selector=filter_conditions)
```
- This proves the infrastructure exists!
- It's just not being used for queries

**4. Metadata Discrepancy**
- Code sets: `doc.metadata['file_path']`
- Qdrant index created on: `fpath`
- This suggests LangChain's `QdrantVectorStore` wrapper maps field names

## Problems Identified:

1. ❌ **No chunk-level metadata** (position, page number, total chunks)
2. ❌ **No file type filtering** (can't search only `.py` or only `.pdf`)
3. ❌ **No temporal metadata** (can't filter by modification date)
4. ❌ **No directory/path filtering** (can't search specific folders)
5. ❌ **MCP server strips metadata** (links are commented out!)
6. ❌ **No content classification** (code vs docs vs data)

## Detailed Findings

### A. Current Metadata Schema

**What's Stored:**
```python
{
    "file_path": "/usr/src/app/local_files/chunker.py",  # Only explicit field
    # LangChain may add:
    # "source": "...",  # (possibly redundant with file_path)
}
```

**Qdrant Collection Configuration:**
```python
# Only ONE payload index exists:
self.qdrant.create_payload_index(
    collection_name="mnm_storage",
    field_name="fpath",  # Note: 'fpath' not 'file_path'
    field_schema="keyword"
)
```

**Chunking Configuration:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
# No chunk metadata added!
```

### B. Query Pipeline Analysis

**Current Flow:**
```
MCP Client
  → minima-query tool
  → MCP Server (server.py)
  → HTTP POST to http://localhost:8001/query
  → indexer.find(query)
  → Pure similarity search (no filters)
  → Post-process: extract file_path for links
  → Return: text only (links commented out in MCP server)
```

**Missed Opportunity:**
```python
# Could be doing this:
found = self.document_store.search(
    query,
    search_type="similarity",
    filter=Filter(must=[
        FieldCondition(key="file_ext", match=MatchValue(value=".py"))
    ])
)
```

### C. MCP Server Issues

**Current Code:**
```python
# minima/mcp-server/src/minima/server.py
output = await request_data(context)
output = output['result']['output']
#links = output['result']['links']  # COMMENTED OUT!
result.append(TextContent(type="text", text=output))
```

**Impact:**
- Users see only concatenated text
- No source attribution
- No way to trace answers back to files

## Proposed Optimizations

### Option 1: Rich Metadata Enhancement (Recommended)

**Enhanced Metadata Schema:**
```python
metadata = {
    # Current
    'file_path': '/home/lostica/projects/mcp_rag/test_docs/chunker.py',

    # File-level metadata
    'file_name': 'chunker.py',
    'file_ext': '.py',
    'file_type': 'code',  # code, doc, data, spreadsheet, presentation
    'directory': '/home/lostica/projects/mcp_rag/test_docs',
    'relative_path': 'test_docs/chunker.py',
    'modified_time': '2025-01-15T10:30:00Z',
    'file_size': 15360,  # bytes

    # Chunk-level metadata
    'chunk_index': 0,
    'total_chunks': 5,
    'chunk_start_char': 0,
    'chunk_end_char': 500,

    # Content-specific (for PDFs)
    'page_number': 3,
    'page_count': 25,

    # Code-specific (for .py, .js, etc.)
    'language': 'python',
    'has_classes': True,
    'has_functions': True,

    # Custom tags (user-defined)
    'tags': ['indexing', 'chunking', 'rag'],
    'priority': 'high',
    'category': 'core',
}
```

**Implementation:**
```python
# In indexer.py _process_file method
import os
from datetime import datetime

def _enrich_metadata(self, doc, loader, chunk_index, total_chunks):
    """Add rich metadata to document chunks."""
    file_path = loader.file_path
    file_ext = os.path.splitext(file_path)[1]

    # File-level
    doc.metadata.update({
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_ext': file_ext,
        'file_type': self._classify_file_type(file_ext),
        'directory': os.path.dirname(file_path),
        'modified_time': datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).isoformat(),
        'file_size': os.path.getsize(file_path),

        # Chunk-level
        'chunk_index': chunk_index,
        'total_chunks': total_chunks,
    })

    # Content-specific
    if file_ext == '.pdf':
        doc.metadata['page_number'] = doc.metadata.get('page', None)

    if file_ext in ['.py', '.js', '.ts', '.cpp', '.h']:
        doc.metadata['language'] = self._detect_language(file_ext)

def _classify_file_type(self, ext: str) -> str:
    """Classify file into broad categories."""
    type_map = {
        'code': ['.py', '.js', '.ts', '.cpp', '.h', '.java', '.go'],
        'doc': ['.md', '.txt', '.pdf', '.docx'],
        'data': ['.csv', '.json', '.xml', '.yaml'],
        'spreadsheet': ['.xlsx', '.xls'],
        'presentation': ['.pptx', '.ppt'],
    }
    for file_type, extensions in type_map.items():
        if ext in extensions:
            return file_type
    return 'other'
```

**Create Additional Qdrant Indexes:**
```python
def _setup_collection(self):
    # Existing code...

    # Create indexes for fast filtering
    indexes_to_create = [
        'metadata.file_ext',
        'metadata.file_type',
        'metadata.directory',
        'metadata.language',
        'metadata.tags',
    ]

    for field_name in indexes_to_create:
        try:
            self.qdrant.create_payload_index(
                collection_name=self.config.QDRANT_COLLECTION,
                field_name=field_name,
                field_schema="keyword"
            )
        except Exception as e:
            logger.warning(f"Could not create index for {field_name}: {e}")

    # Numeric indexes for range queries
    self.qdrant.create_payload_index(
        collection_name=self.config.QDRANT_COLLECTION,
        field_name="metadata.chunk_index",
        field_schema="integer"
    )
```

**Enhanced Query API:**
```python
def find(
    self,
    query: str,
    file_types: List[str] = None,
    file_extensions: List[str] = None,
    directories: List[str] = None,
    languages: List[str] = None,
    tags: List[str] = None,
    modified_after: str = None,
    max_chunk_index: int = None
) -> Dict[str, any]:
    """
    Enhanced search with metadata filtering.

    Args:
        query: Search query string
        file_types: Filter by file type (code, doc, data, etc.)
        file_extensions: Filter by extension (.py, .pdf, etc.)
        directories: Filter by directory path
        languages: Filter by programming language
        tags: Filter by custom tags
        modified_after: ISO timestamp - files modified after this date
        max_chunk_index: Only return first N chunks (for focusing on file starts)
    """
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
        # Use prefix matching for directories
        should_conditions = [
            FieldCondition(
                key="metadata.directory",
                match=MatchValue(value=dir_path)
            )
            for dir_path in directories
        ]
        must_conditions.append(
            Filter(should=should_conditions, min_should_match=1)
        )

    if languages:
        must_conditions.append(
            FieldCondition(
                key="metadata.language",
                match=MatchAny(any=languages)
            )
        )

    if tags:
        for tag in tags:
            must_conditions.append(
                FieldCondition(
                    key="metadata.tags",
                    match=MatchValue(value=tag)
                )
            )

    if modified_after:
        must_conditions.append(
            FieldCondition(
                key="metadata.modified_time",
                range=Range(gte=modified_after)
            )
        )

    if max_chunk_index is not None:
        must_conditions.append(
            FieldCondition(
                key="metadata.chunk_index",
                range=Range(lte=max_chunk_index)
            )
        )

    # Create filter
    filter_obj = Filter(must=must_conditions) if must_conditions else None

    # Search with filters
    found = self.document_store.search(
        query,
        search_type="similarity",
        filter=filter_obj
    )

    # Process results with enhanced metadata
    return self._format_results(found)

def _format_results(self, found) -> Dict[str, any]:
    """Format results with metadata information."""
    if not found:
        return {"links": [], "output": "", "metadata": []}

    links = []
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
            "chunk": f"{meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', '?')}",
            "modified": meta.get("modified_time"),
        }

        links.append(f"file://{path}")
        results.append(item.page_content)
        metadata_info.append(file_info)

    return {
        "links": list(set(links)),
        "output": ". ".join(results),
        "metadata": metadata_info,
        "total_results": len(found)
    }
```

**Update HTTP API Endpoint:**
```python
# In FastAPI app
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    file_types: Optional[List[str]] = None
    file_extensions: Optional[List[str]] = None
    directories: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    modified_after: Optional[str] = None
    max_chunk_index: Optional[int] = None

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = indexer.find(
        query=request.query,
        file_types=request.file_types,
        file_extensions=request.file_extensions,
        directories=request.directories,
        languages=request.languages,
        tags=request.tags,
        modified_after=request.modified_after,
        max_chunk_index=request.max_chunk_index
    )
    return {"result": result}
```

### Option 2: Hybrid Search Enhancement

Combine vector similarity with metadata-based scoring:

```python
def hybrid_search(self, query: str, boost_recent: float = 0.1, boost_early_chunks: float = 0.05):
    """
    Hybrid search combining:
    - Vector similarity (primary)
    - Recency boost (newer files ranked higher)
    - Position boost (earlier chunks ranked higher)
    """
    # Get similarity results
    results = self.document_store.search(query, search_type="similarity", k=20)

    # Re-rank with metadata
    from datetime import datetime
    now = datetime.now()

    for result in results:
        meta = result.metadata

        # Recency score
        if 'modified_time' in meta:
            mod_time = datetime.fromisoformat(meta['modified_time'])
            days_old = (now - mod_time).days
            recency_score = 1.0 / (1.0 + days_old / 30)  # Decay over ~30 days
            result.score += boost_recent * recency_score

        # Early chunk bonus
        if 'chunk_index' in meta and 'total_chunks' in meta:
            position_ratio = meta['chunk_index'] / max(meta['total_chunks'], 1)
            early_bonus = 1.0 - position_ratio
            result.score += boost_early_chunks * early_bonus

    # Re-sort by adjusted scores
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:10]
```

### Option 3: MCP Server Enhancement

**Restore Links and Add Metadata:**
```python
# minima/mcp-server/src/minima/server.py

@server.call_tool()
async def call_tool(name, arguments: dict) -> list[TextContent]:
    # ... validation ...

    output = await request_data(context)
    if "error" in output:
        logging.error(output["error"])
        raise McpError(INTERNAL_ERROR, output["error"])

    result_data = output['result']

    # Format response with metadata
    response_parts = []

    # Add sources if available
    if 'links' in result_data and result_data['links']:
        links_formatted = "\n".join([f"- {link}" for link in result_data['links']])
        response_parts.append(f"**Sources ({len(result_data['links'])}):**\n{links_formatted}")

    # Add metadata if available
    if 'metadata' in result_data and result_data['metadata']:
        metadata_formatted = "\n".join([
            f"- {m['file_name']} (chunk {m['chunk']}, {m['file_type']})"
            for m in result_data['metadata'][:5]  # Top 5
        ])
        response_parts.append(f"\n**Chunks:**\n{metadata_formatted}")

    # Add answer
    response_parts.append(f"\n**Answer:**\n{result_data['output']}")

    full_response = "\n".join(response_parts)

    return [TextContent(type="text", text=full_response)]
```

**Enhanced MCP Tool with Filter Options:**
```python
tools = [
    types.Tool(
        name="minima-query",
        description="Query local documents with optional filtering",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The search query"
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by file type: code, doc, data, etc."
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by extension: .py, .pdf, .md, etc."
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by language: python, javascript, etc."
                },
            },
            "required": ["text"],
        },
    )
]

# Update requestor to pass filters
async def request_data(context: str, filters: dict = None):
    payload = {"query": context}
    if filters:
        payload.update(filters)

    async with aiohttp.ClientSession() as session:
        async with session.post(REQUEST_DATA_URL, json=payload) as resp:
            return await resp.json()
```

## Quick Wins (Immediate Impact)

### 1. Uncomment Links in MCP Server (5 minutes)
**File:** `minima/mcp-server/src/minima/server.py`

**Change:**
```python
# Before:
output = output['result']['output']
#links = output['result']['links']  # COMMENTED OUT!

# After:
result_data = output['result']
output_text = result_data['output']
links = result_data.get('links', [])

# Format with sources
if links:
    links_formatted = "\n".join([f"- {link}" for link in links])
    full_response = f"**Sources:**\n{links_formatted}\n\n**Answer:**\n{output_text}"
else:
    full_response = output_text

return [TextContent(type="text", text=full_response)]
```

### 2. Add Basic File Metadata (15 minutes)
**File:** `minima/indexer/indexer.py`

**Add to `_process_file` method:**
```python
import os
from datetime import datetime

def _process_file(self, loader) -> List[str]:
    try:
        documents = loader.load_and_split(self.text_splitter)
        if not documents:
            return []

        file_path = loader.file_path
        file_ext = os.path.splitext(file_path)[1]
        total_chunks = len(documents)

        # Enhanced metadata
        for idx, doc in enumerate(documents):
            doc.metadata.update({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_ext': file_ext,
                'chunk_index': idx,
                'total_chunks': total_chunks,
            })

        # Rest of the method...
```

### 3. Create Payload Index for Extensions (5 minutes)
**File:** `minima/indexer/indexer.py`

**Add to `_setup_collection` method:**
```python
def _setup_collection(self) -> QdrantVectorStore:
    # ... existing code ...

    # Add extension index
    try:
        self.qdrant.create_payload_index(
            collection_name=self.config.QDRANT_COLLECTION,
            field_name="metadata.file_ext",
            field_schema="keyword"
        )
    except Exception as e:
        logger.info(f"Extension index may already exist: {e}")

    return QdrantVectorStore(...)
```

### 4. Test Filtering (5 minutes)
```python
# Test script
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", 6333)

# Search only Python files
results = client.search(
    collection_name="mnm_storage",
    query_vector=[0.1] * 768,  # Dummy vector
    query_filter=Filter(
        must=[
            FieldCondition(
                key="metadata.file_ext",
                match=MatchValue(value=".py")
            )
        ]
    ),
    limit=5
)

print(f"Found {len(results)} Python file chunks")
for r in results:
    print(f"  - {r.payload.get('metadata', {}).get('file_name')}")
```

## Implementation Roadmap

### Phase 1: Foundation (1 hour)
- [ ] Uncomment links in MCP server
- [ ] Add basic file metadata (name, ext, chunk index)
- [ ] Create payload indexes for file_ext
- [ ] Test basic filtering

### Phase 2: Rich Metadata (2-3 hours)
- [ ] Add file type classification
- [ ] Add temporal metadata (modified time)
- [ ] Add directory metadata
- [ ] Create additional payload indexes
- [ ] Test metadata enrichment

### Phase 3: Query Enhancement (2-3 hours)
- [ ] Extend `find()` method with filter parameters
- [ ] Update FastAPI endpoint with filter support
- [ ] Add filter validation
- [ ] Test query filtering

### Phase 4: MCP Integration (1-2 hours)
- [ ] Update MCP tool schema with filter options
- [ ] Update requestor to pass filters
- [ ] Format responses with metadata
- [ ] Test end-to-end filtering from Claude Code

### Phase 5: Advanced Features (2-4 hours)
- [ ] Implement hybrid search with metadata scoring
- [ ] Add custom tagging support
- [ ] Add content-specific metadata (page numbers, language detection)
- [ ] Performance optimization

## Expected Benefits

### Performance:
- **Faster queries**: Pre-filtering in Qdrant vs post-filtering in Python
- **Reduced token usage**: Filter before retrieval vs retrieve-all-then-filter
- **Better recall**: Target specific document types

### UX:
- **Source attribution**: Users see where answers come from
- **Targeted searches**: "Search only Python files for error handling"
- **Temporal filtering**: "Show recent changes to authentication code"

### Developer Experience:
- **Debugging**: Metadata shows which chunks contributed
- **Control**: Fine-grained query control
- **Transparency**: Clear source tracking

## Example Use Cases

### Use Case 1: Search Only Code Files
```python
# Claude Code query:
"Use minima-query to search for authentication logic in Python files only"

# MCP request:
{
    "text": "authentication logic",
    "file_extensions": [".py"]
}

# Result: Only Python files searched, faster and more relevant
```

### Use Case 2: Recent Documentation Changes
```python
# Claude Code query:
"Use minima-query to find recent changes to API documentation"

# MCP request:
{
    "text": "API documentation",
    "file_types": ["doc"],
    "modified_after": "2025-01-01T00:00:00Z"
}

# Result: Only recent markdown/doc files
```

### Use Case 3: File Introductions Only
```python
# Claude Code query:
"Use minima-query to find overview of chunking implementation (first chunk only)"

# MCP request:
{
    "text": "chunking implementation",
    "max_chunk_index": 0
}

# Result: Only first chunks (file introductions/docstrings)
```

## Migration Notes

### Data Re-indexing Required:
- Metadata changes require re-indexing existing vectors
- Options:
  1. **Clean slate**: `docker compose down -v && rm -rf qdrant_data indexer_data`
  2. **Gradual**: New metadata added only to new/modified files
  3. **Forced re-index**: Delete all, trigger full re-crawl

### Backward Compatibility:
- Enhanced query API should have optional parameters (backward compatible)
- Old queries without filters work as before
- MCP tool can remain backward compatible with optional filter fields

## Testing Checklist

- [ ] Metadata correctly added during indexing
- [ ] Payload indexes created without errors
- [ ] Filter by file extension works
- [ ] Filter by file type works
- [ ] Filter by directory works
- [ ] Multiple filters combine correctly (AND logic)
- [ ] Links appear in MCP responses
- [ ] Metadata displayed in MCP responses
- [ ] Query without filters still works (backward compat)
- [ ] Performance: filtered queries faster than unfiltered

## References

### Qdrant Documentation:
- Payload filtering: https://qdrant.tech/documentation/concepts/filtering/
- Payload indexes: https://qdrant.tech/documentation/concepts/indexing/
- Search with filters: https://qdrant.tech/documentation/concepts/search/

### LangChain Documentation:
- QdrantVectorStore: https://python.langchain.com/docs/integrations/vectorstores/qdrant
- Metadata filtering: https://python.langchain.com/docs/modules/data_connection/vectorstores/

### Current Codebase:
- Indexer: `/home/lostica/projects/mcp_rag/minima/indexer/indexer.py`
- MCP Server: `/home/lostica/projects/mcp_rag/minima/mcp-server/src/minima/server.py`
- API Config: `/home/lostica/projects/mcp_rag/minima/.env`

## Conclusion

The current Minima implementation has solid infrastructure but underutilizes Qdrant's metadata capabilities. By implementing the proposed enhancements, we can achieve:

1. **Better query relevance** through pre-filtering
2. **Faster queries** by reducing search space
3. **Better UX** with source attribution
4. **More control** for users and developers

**Recommended Start:** Implement Phase 1 (Quick Wins) immediately, then iterate based on usage patterns.
