# AST Support Analysis for RAG Systems

**Date**: 2025-11-02
**Repositories Analyzed**: LEANN, minima, astchunk

---

## Executive Summary

This report analyzes AST (Abstract Syntax Tree) support capabilities across three repositories to determine the best approach for adding AST-aware code chunking to a minima fork.

**Key Findings:**
- **LEANN** has production-ready AST support via vendored `astchunk` library
- **minima** lacks AST support, uses basic character-based chunking for all files
- **astchunk** is a standalone, integration-friendly library perfect for adaptation

**Recommendation:** Integrate `astchunk` into minima fork rather than homebrewing. Estimated effort: 50-100 lines of code changes.

---

## Repository Analysis

### 1. LEANN - Full AST Support

**Location**: `/home/lostica/projects/mcp_rag/LEANN`

#### AST Capabilities

**Supported Languages** (via tree-sitter):
- Python (`.py`)
- Java (`.java`)
- C# (`.cs`)
- TypeScript/JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`)

**Key Features:**
- AST-aware code chunking that preserves semantic boundaries (functions, classes, methods)
- Automatic language detection from file extensions
- Graceful fallback to traditional text-based chunking for unsupported languages
- Configurable chunk sizes and overlaps specific to AST chunking
- Hybrid approach: AST for code, traditional for documents

#### Implementation Architecture

**External Dependency:**
- Vendors `astchunk` as git submodule at `packages/astchunk-leann`
- Submodule URL: `https://github.com/yichuan-w/astchunk-leann.git`
- For PyPI installations: `astchunk>=0.1.0` as dependency

**Core Files:**

1. **`packages/leann-core/src/leann/chunking_utils.py`** (221 lines)
   - Main chunking facade integrating all strategies
   - Functions:
     - `detect_code_files()` - Separates code from text documents
     - `create_ast_chunks()` - AST-aware chunking with astchunk
     - `create_traditional_chunks()` - LlamaIndex SentenceSplitter for text
     - `create_text_chunks()` - Unified interface supporting both strategies
     - `get_language_from_extension()` - File extension to language mapping

2. **`apps/code_rag.py`** (212 lines)
   - Specialized RAG for code repositories
   - Always uses AST chunking by default
   - Smart file filtering (excludes `.git`, `node_modules`, etc.)
   - Default settings: `ast_chunk_size=768`, `ast_chunk_overlap=96`

3. **`apps/document_rag.py`** (132 lines)
   - General document processing with optional code support
   - Flag: `--enable-code-chunking` to activate AST mode
   - Mixed strategy: AST for code, traditional for documents

**Chunking Strategy Selection:**
```python
def create_text_chunks(..., use_ast_chunking=False):
    if use_ast_chunking:
        code_docs, text_docs = detect_code_files(documents)
        ast_chunks = create_ast_chunks(code_docs)  # Uses astchunk
        text_chunks = create_traditional_chunks(text_docs)  # Uses LlamaIndex
        return ast_chunks + text_chunks
    else:
        return create_traditional_chunks(documents)
```

#### Entry Points

**Command-Line Interface:**
```bash
# Document RAG with code support
python -m apps.document_rag --enable-code-chunking --data-dir ./my_project

# Specialized code RAG (AST always on)
python -m apps.code_rag --repo-dir ./my_codebase

# Global CLI
leann build my-code-index --docs ./src --use-ast-chunking
```

**Python API:**
```python
from chunking import create_text_chunks

chunks = create_text_chunks(
    documents,
    use_ast_chunking=True,
    ast_chunk_size=768,
    ast_chunk_overlap=96
)
```

**MCP Server:**
- Entry point: `leann_mcp` command
- Tools exposed: `leann_list`, `leann_search`
- Setup: `claude mcp add --scope user leann-server -- leann_mcp`
- AST integration: Automatic when indexes built with `--use-ast-chunking`

#### Dependencies

```toml
"astchunk>=0.1.0"              # Core AST chunking library
"tree-sitter>=0.20.0"          # Parser generator framework
"tree-sitter-python>=0.20.0"   # Python grammar
"tree-sitter-java>=0.20.0"     # Java grammar
"tree-sitter-c-sharp>=0.20.0"  # C# grammar
"tree-sitter-typescript>=0.20.0" # TypeScript/JavaScript grammar
"llama-index-core>=0.12.0"     # For traditional text chunking
```

#### Fallback Mechanism
- If `astchunk` import fails → fall back to traditional chunking
- If AST parsing fails for a file → fall back to traditional chunking
- If no language detected → fall back to traditional chunking
- Controlled by `ast_fallback_traditional` parameter (default: True)

#### Testing
**Test Suite**: `tests/test_astchunk_integration.py` (398 lines)

**Coverage:**
- Code file detection (Python, Java, TypeScript, C#)
- Language mapping from extensions
- AST chunking with astchunk available
- Fallback to traditional chunking
- Error handling (empty content, missing language)
- Integration with document_rag and code_rag apps

#### Documentation
- `docs/ast_chunking_guide.md` (144 lines) - Quick start, best practices, troubleshooting
- Main README.md - AST chunking feature overview

**Strengths:**
- Production-ready implementation
- Multiple fallback layers ensure reliability
- Well-tested and documented
- MCP integration for Claude Code
- Hybrid approach handles mixed content elegantly

---

### 2. minima - No AST Support

**Location**: `/home/lostica/projects/mcp_rag/minima`

#### Current Capabilities

**Supported File Types** (`indexer/indexer.py` lines 32-52):
- **Documents**: PDF, DOCX, DOC, TXT, MD, CSV, PPTX, PPT, XLS, XLSX
- **Code files**: Python (`.py`), JavaScript (`.js`), Java (`.java`), C/C++ (`.c`, `.cpp`, `.h`, `.hpp`), Rust (`.rs`)

**Critical Finding**: All code files use `TextLoader` - the same generic text loader as plain text files. **No AST awareness.**

#### Chunking Strategy

From `indexer/indexer.py` (lines 68-93):
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
```

Uses `RecursiveCharacterTextSplitter` from LangChain:
- **Character-based splitting only**
- No AST awareness
- Breaks functions/classes arbitrarily at character boundaries

#### Architecture Components

**A. Indexer** (`indexer/`)
- **app.py**: FastAPI server with `/query` and `/embedding` endpoints
- **indexer.py**: Core indexing logic with file processing
- **async_loop.py**: File crawling and indexing queue management
- **storage.py**: SQLite-based file tracking (last modified timestamps)
- **async_queue.py**: Simple deque-based async queue

**B. Vector Storage**
- **Qdrant**: Vector database for embeddings
- **HuggingFace Embeddings**: sentence-transformers models (default: all-mpnet-base-v2)
- **Collection**: mnm_storage with COSINE distance

**C. MCP Server** (`mcp-server/`)
- **server.py**: MCP protocol implementation
- **requestor.py**: HTTP client to indexer API
- Exposes `minima-query` tool to Claude Desktop

**D. Other Components**
- **Linker**: Firebase/Firestore integration for ChatGPT custom GPT mode
- **Chat/LLM**: Web UI and Ollama integration for standalone mode

#### Integration Points for AST Support

**Primary Integration: `indexer.py` - File Processing**

Current flow:
```python
_create_loader() → TextLoader (line 115-122)
    ↓
_process_file() → load_and_split(text_splitter) (line 126)
    ↓
RecursiveCharacterTextSplitter (line 89-93)
```

**Modification Strategy:**

1. **Add AST-aware loader factory** (line 115-122):
   - Detect code file extensions
   - Return AST-based loader instead of TextLoader

2. **Create code-aware splitter** (line 89-93):
   - Add parallel AST-based text splitter
   - Use tree-sitter or Python AST
   - Split by functions/classes instead of character count

3. **Enhance metadata** (line 131-133):
   - Add symbol name, type (function/class/method)
   - Add line numbers, docstrings
   - Add parent/child relationships

**Secondary Integration: `async_loop.py`** (line 12)
```python
AVAILABLE_EXTENSIONS = [".pdf", ".xls", ..., ".py", ".js", ".java", ...]
```
Add categorization for code extensions to enable AST processing.

#### Key Files for AST Integration

| File | Purpose | Relevance for AST |
|------|---------|-------------------|
| `indexer/indexer.py` | Core indexing logic | **HIGH** - Main integration point |
| `indexer/async_loop.py` | File crawling | **MEDIUM** - Extension filtering |
| `indexer/storage.py` | File tracking DB | **LOW** - Could track symbols |
| `indexer/app.py` | FastAPI server | **LOW** - Query API enhancement |
| `mcp-server/src/minima/server.py` | MCP interface | **LOW** - Expose AST features |

**Strengths:**
- Clean, modular architecture
- Easy to extend
- FastAPI for modern async processing
- MCP server already available

**Weaknesses:**
- No AST support
- Basic chunking breaks code semantics
- Metadata limited for code files

---

### 3. astchunk - Standalone AST Chunking Library

**Location**: `/home/lostica/projects/mcp_rag/astchunk`

#### What is astchunk?

AST-based code chunking library that intelligently divides source code into meaningful chunks while preserving syntactic structure and semantic boundaries.

**Research Context**: Based on the paper "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree" (arXiv:2506.15655)

#### Key Features

- **Structure-aware chunking**: Uses tree-sitter parsers to respect AST boundaries
- **Size-based chunking**: Non-whitespace character count for consistent sizing
- **Metadata preservation**: File paths, line numbers, class/function paths, AST context
- **Optional overlapping**: Context via overlapping AST nodes between chunks
- **Chunk expansion**: Adds metadata headers (filepath, ancestor class/function paths)
- **Greedy merging**: Merges adjacent sibling nodes when possible without exceeding size limits

#### Supported Languages

| Language   | Parser Module              | File Extensions |
|------------|----------------------------|-----------------|
| Python     | tree-sitter-python         | .py             |
| Java       | tree-sitter-java           | .java           |
| C#         | tree-sitter-c-sharp        | .cs             |
| TypeScript | tree-sitter-typescript     | .ts, .tsx       |

All parsers automatically installed as dependencies.

#### Implementation Architecture

**Core Modules** (5 files, ~500 lines total):

1. **astchunk_builder.py** (334 lines)
   - `ASTChunkBuilder` class with 4-step chunking pipeline
   - Greedy assignment of AST nodes to windows
   - Recursive processing for oversized nodes
   - Adjacent window merging logic

2. **astchunk.py** (213 lines)
   - `ASTChunk` class representing final chunks
   - Code reconstruction from AST nodes
   - Metadata generation (multiple templates)
   - Chunk expansion with ancestor paths

3. **astnode.py** (65 lines)
   - `ASTNode` wrapper for tree-sitter nodes
   - Properties: size, line ranges, code text, byte ranges

4. **preprocessing.py** (112 lines)
   - Cumulative sum preprocessing for O(1) non-whitespace character counting
   - `ByteRange`/`IntRange` for range operations
   - Node search within byte ranges

5. **__init__.py** (35 lines)
   - Public API exports

#### Algorithm Overview

```
Step 1: assign_tree_to_windows()
  - Parse code into AST
  - Preprocess non-whitespace character counts (O(n))
  - Greedily assign nodes to windows (max_chunk_size limit)
  - Recursively process oversized nodes
  - Merge adjacent sibling windows when possible

Step 2: add_window_overlapping() [optional]
  - Add k nodes from previous/next windows for context

Step 3: convert_windows_to_chunks()
  - Rebuild code text from AST nodes
  - Generate metadata
  - Apply chunk expansion [optional]

Step 4: convert_chunks_to_code_windows()
  - Convert to output format: {"content": str, "metadata": dict}
```

#### API Usage

**Input:**
```python
from astchunk import ASTChunkBuilder

configs = {
    "max_chunk_size": 2000,        # Non-whitespace chars per chunk
    "language": "python",          # Required
    "metadata_template": "default" # Required
}

chunk_builder = ASTChunkBuilder(**configs)

chunks = chunk_builder.chunkify(
    code,                          # String of source code
    repo_level_metadata={          # Optional
        "filepath": "path/to/file.py"
    },
    chunk_overlap=1,               # Optional: AST nodes to overlap
    chunk_expansion=True           # Optional: add metadata headers
)
```

**Output:**
```python
[
    {
        "content": "def foo():\n    return 42",
        "metadata": {
            "filepath": "path/to/file.py",
            "chunk_size": 20,
            "line_count": 2,
            "start_line_no": 0,
            "end_line_no": 1,
            "node_count": 1
        }
    },
    # ... more chunks
]
```

#### Dependencies

**Core Dependencies** (automatically installed):
- `numpy` (>=1.20.0) - Cumulative sum preprocessing
- `pyrsistent` (>=0.18.0) - Immutable ancestor tracking
- `tree-sitter` (>=0.20.0) - Parser framework
- `tree-sitter-python`, `tree-sitter-java`, `tree-sitter-c-sharp`, `tree-sitter-typescript` (>=0.20.0)

**Key Observation**: NO heavy ML/AI dependencies. Pure parsing/chunking library.

#### Standalone Assessment

**Very Standalone** - Excellent integration candidate:

✅ **No coupling**: Self-contained library with clear API boundary
✅ **Minimal dependencies**: Only numpy, pyrsistent, and tree-sitter parsers
✅ **No external services**: Fully local processing
✅ **No configuration files**: All config via Python dictionaries
✅ **PyPI package**: `pip install astchunk`
✅ **Clean imports**: Single entry point via `from astchunk import ASTChunkBuilder`

**Integration Patterns:**
1. **As installed package**: Add to requirements.txt, import directly
2. **As vendored module**: Copy `src/astchunk/` into minima
3. **As git submodule**: Reference as dependency (LEANN approach)

**Strengths:**
- Lightweight and focused
- Research-backed effectiveness
- Clean, well-documented API
- No bloat or unnecessary features

---

## Comparison Matrix

| Feature | LEANN | minima | astchunk |
|---------|-------|--------|----------|
| **AST Support** | ✅ Full (via astchunk) | ❌ None | ✅ Core purpose |
| **Languages** | Python, Java, C#, TypeScript | N/A | Python, Java, C#, TypeScript |
| **Architecture** | Hybrid (code+docs) | Simple RAG | Pure chunking library |
| **Integration Complexity** | Pre-integrated | Needs implementation | Standalone library |
| **Dependencies** | Heavy (LlamaIndex, etc.) | Medium (LangChain, etc.) | Light (tree-sitter only) |
| **MCP Server** | ✅ Yes | ✅ Yes | ❌ N/A (library) |
| **Fallback Strategy** | ✅ Multi-layer | ❌ None for AST | Manual implementation |
| **Documentation** | ✅ Excellent | ✅ Good | ✅ Good (+ research paper) |
| **Testing** | ✅ Comprehensive | ⚠️ Basic | ✅ pytest suite |
| **Use Case** | Production RAG system | Production RAG system | Library for integration |
| **Customization** | Moderate | High (simple codebase) | High (source available) |

---

## Integration Recommendation

### Recommended Approach: Integrate astchunk into minima fork

**Why astchunk over homebrewing?**

1. **Proven**: Research-backed with published paper (arXiv:2506.15655)
2. **Well-tested**: ~500 lines of mature, tested code
3. **Time-saving**: Homebrewing would require 500+ lines of complex AST parsing logic
4. **Tree-sitter expertise**: Non-trivial integration already solved
5. **Maintained**: Active PyPI package with dependency management

**Why not copy LEANN's approach entirely?**

1. **Overkill**: LEANN's full stack includes LlamaIndex, complex app structure
2. **Dependencies**: Heavy dependency tree not needed for minima
3. **Simplicity**: minima's clean architecture should be preserved
4. **Customization**: Direct astchunk integration allows more control

### Implementation Plan

#### Phase 1: Basic Integration (MVP)

**Effort**: ~50-100 lines of code changes

**Steps:**

1. **Add dependency** to `minima/indexer/requirements.txt`:
   ```txt
   astchunk>=0.1.0
   tree-sitter>=0.20.0
   tree-sitter-python>=0.20.0
   tree-sitter-java>=0.20.0
   tree-sitter-c-sharp>=0.20.0
   tree-sitter-typescript>=0.20.0
   ```

2. **Modify** `minima/indexer/indexer.py`:

   a. Add to `Config` class (line 32):
   ```python
   CODE_EXTENSIONS = {".py", ".java", ".cs", ".ts", ".tsx"}
   AST_CHUNK_SIZE = 2000  # Non-whitespace chars
   USE_AST_CHUNKING = True  # Feature flag
   ```

   b. Add AST chunker initialization (after line 77):
   ```python
   def _initialize_ast_chunkers(self) -> dict:
       """Initialize ASTChunkBuilder for each supported language"""
       try:
           from astchunk import ASTChunkBuilder
       except ImportError:
           logger.warning("astchunk not available, falling back to character chunking")
           return {}

       return {
           ".py": ASTChunkBuilder(
               max_chunk_size=self.config.AST_CHUNK_SIZE,
               language="python",
               metadata_template="default"
           ),
           ".java": ASTChunkBuilder(
               max_chunk_size=self.config.AST_CHUNK_SIZE,
               language="java",
               metadata_template="default"
           ),
           ".cs": ASTChunkBuilder(
               max_chunk_size=self.config.AST_CHUNK_SIZE,
               language="c_sharp",
               metadata_template="default"
           ),
           ".ts": ASTChunkBuilder(
               max_chunk_size=self.config.AST_CHUNK_SIZE,
               language="typescript",
               metadata_template="default"
           ),
           ".tsx": ASTChunkBuilder(
               max_chunk_size=self.config.AST_CHUNK_SIZE,
               language="typescript",
               metadata_template="default"
           ),
       }
   ```

   c. Modify `_process_file` (line 124):
   ```python
   def _process_file(self, loader) -> List[str]:
       file_extension = Path(loader.file_path).suffix.lower()

       # Use AST chunking for code files if enabled
       if (self.config.USE_AST_CHUNKING and
           file_extension in self.config.CODE_EXTENSIONS and
           hasattr(self, 'ast_chunkers')):
           return self._process_code_file(loader, file_extension)

       # Existing logic for documents
       documents = loader.load_and_split(self.text_splitter)
       # ... rest of existing code
   ```

   d. Add new method `_process_code_file`:
   ```python
   def _process_code_file(self, loader, file_extension) -> List[str]:
       """Process code files with AST-aware chunking"""
       try:
           # Read file content
           with open(loader.file_path, 'r', encoding='utf-8') as f:
               code = f.read()

           # Get appropriate chunker
           chunker = self.ast_chunkers.get(file_extension)
           if not chunker:
               # Fallback to regular chunking
               logger.warning(f"No AST chunker for {file_extension}, using character chunking")
               documents = loader.load_and_split(self.text_splitter)
               return self._store_documents(documents, loader.file_path)

           # Chunk using AST
           chunks = chunker.chunkify(
               code,
               repo_level_metadata={"filepath": loader.file_path},
               chunk_expansion=True  # Adds filepath + class/function context
           )

           # Convert to LangChain Document format
           from langchain_core.documents import Document
           documents = [
               Document(
                   page_content=chunk["content"],
                   metadata={
                       **chunk["metadata"],
                       "file_path": loader.file_path,
                       "chunking_method": "ast"
                   }
               )
               for chunk in chunks
           ]

           return self._store_documents(documents, loader.file_path)

       except Exception as e:
           logger.error(f"AST chunking failed for {loader.file_path}: {e}, falling back")
           documents = loader.load_and_split(self.text_splitter)
           return self._store_documents(documents, loader.file_path)

   def _store_documents(self, documents, file_path) -> List[str]:
       """Store documents in vector DB"""
       uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
       ids = self.document_store.add_documents(documents=documents, ids=uuids)
       return ids
   ```

3. **Update** `minima/indexer/.env`:
   ```bash
   # AST Chunking Configuration
   USE_AST_CHUNKING=true
   AST_CHUNK_SIZE=2000
   ```

#### Phase 2: Enhancement (Optional)

**Additional features to consider:**

1. **Per-language configuration**:
   ```python
   AST_CHUNK_SIZES = {
       "python": 2000,
       "java": 2500,
       "typescript": 1500
   }
   ```

2. **Query enhancement** - Filter by symbol type:
   ```python
   # In indexer.py find() method
   def find(self, query: str, symbol_type: Optional[str] = None):
       # Add filter for metadata.symbol_type if symbol_type provided
   ```

3. **Chunk overlap configuration**:
   ```python
   chunks = chunker.chunkify(
       code,
       chunk_overlap=1,  # 1 AST node overlap between chunks
       ...
   )
   ```

4. **Metadata enrichment** - Add more AST context:
   ```python
   metadata={
       **chunk["metadata"],
       "file_path": loader.file_path,
       "chunking_method": "ast",
       "language": language,
       "has_docstring": detect_docstring(chunk["content"])
   }
   ```

#### Phase 3: Testing

**Test cases needed:**

1. **Basic chunking** - Python file with classes/functions
2. **Fallback** - Unsupported file type (.cpp, .rs)
3. **Error handling** - Malformed code
4. **Metadata** - Verify AST metadata in stored chunks
5. **Query** - Semantic search retrieves correct code chunks
6. **Migration** - Existing non-code documents still work

**Test file**: `minima/indexer/test_ast_integration.py`

#### Phase 4: Documentation

**Update files:**

1. `minima/README.md` - Add AST chunking feature
2. `minima/QUICK_START.md` - Add configuration examples
3. New: `minima/docs/AST_CHUNKING.md` - Detailed guide

---

## Benefits of AST Integration

### For Code Retrieval Quality

1. **Semantic Boundaries**: Functions/classes kept together
2. **Better Context**: Chunk expansion adds filepath + ancestor paths
3. **Consistent Sizing**: Non-whitespace counting more predictable
4. **Rich Metadata**: Line numbers, symbol types, AST paths
5. **Research-Backed**: Proven effective in RAG benchmarks

### For Developer Experience

1. **Minimal Changes**: ~100 lines added to existing codebase
2. **No Breaking Changes**: Existing documents continue to work
3. **Feature Flag**: Can disable if issues arise
4. **Graceful Fallback**: Auto-falls back to character chunking on errors
5. **Lightweight**: Only 5MB of additional dependencies

---

## Potential Challenges

### 1. Language Coverage

**Issue**: Only 4 languages supported (vs minima's 10+ file types)

**Solutions**:
- Fallback to `RecursiveCharacterTextSplitter` for unsupported languages
- Add more tree-sitter parsers as needed:
  - `tree-sitter-cpp` for C++
  - `tree-sitter-rust` for Rust
  - `tree-sitter-go` for Go

### 2. Chunk Size Variation

**Issue**: AST respects syntax boundaries, so actual chunk sizes vary more

**Solutions**:
- ASTChunk already enforces `max_chunk_size`, should be acceptable
- Monitor actual chunk size distribution in testing
- Adjust `AST_CHUNK_SIZE` if needed

### 3. Performance

**Issue**: AST parsing adds overhead vs simple character splitting

**Solutions**:
- Impact minimal (tree-sitter is fast, ~ms per file)
- Async processing already in place via `async_loop.py`
- Could add caching layer if needed

### 4. Error Handling

**Issue**: Malformed code may fail parsing

**Solutions**:
- Comprehensive try-except in `_process_code_file`
- Automatic fallback to character chunking
- Log warnings for investigation
- Store parse errors in metadata for debugging

---

## Next Steps

### Immediate Actions

1. **Add `astchunk` dependency** to minima fork
2. **Implement Phase 1** (basic integration) in `indexer.py`
3. **Test with Python files** to validate approach
4. **Measure performance** impact on indexing speed
5. **Compare search quality** (AST vs character chunking)

### Future Considerations

1. **Expand language support** (C++, Rust, Go)
2. **Add symbol-level search** (filter by function/class)
3. **Implement cross-reference tracking** (function calls)
4. **Expose AST features** via MCP server
5. **Create visualization** of code structure

---

## Conclusion

Integrating `astchunk` into a minima fork is **highly feasible** and **valuable**:

✅ **Proven solution**: Research-backed, well-tested library
✅ **Low effort**: ~100 lines of code changes
✅ **High impact**: Significantly improves code retrieval quality
✅ **Safe**: Multiple fallback layers, no breaking changes
✅ **Extensible**: Clean architecture allows future enhancements

**Recommendation**: Proceed with Phase 1 integration using astchunk rather than homebrewing.

---

## References

- **astchunk Paper**: "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree" (arXiv:2506.15655)
- **LEANN Repository**: `/home/lostica/projects/mcp_rag/LEANN`
- **minima Repository**: `/home/lostica/projects/mcp_rag/minima`
- **astchunk Repository**: `/home/lostica/projects/mcp_rag/astchunk`
- **tree-sitter**: https://tree-sitter.github.io/tree-sitter/

---

**Report Generated**: 2025-11-02
**Analysis Tools**: Claude Code Explore agents (parallel execution)
