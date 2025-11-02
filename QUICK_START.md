# Minima Quick Start

Quick reference for running your local RAG system with Qdrant + Minima.

---

## First Time Setup

```bash
cd /path/to/minima

# Build and start everything (takes ~5 min first time)
docker compose -f docker-compose-mcp.yml up --build
```

**Wait for:** `"all files indexed"` message in logs.

---

## Initial Indexing

**First time only:** After starting services, wait for indexing to complete.

### Watch the indexing progress:

```bash
# Follow indexer logs
docker compose -f docker-compose-mcp.yml logs -f indexer
```

You'll see:
```
indexer-1  | INFO: Loading embedding model: all-mpnet-base-v2
indexer-1  | INFO: Processing file: /usr/src/app/local_files/chunker.py
indexer-1  | INFO: Successfully processed 15 documents from chunker.py
indexer-1  | INFO: Added 15 vectors. Total: 15
...
indexer-1  | INFO: No files to index. Indexing stopped, all files indexed.
```

### Verify indexing completed:

```bash
# Check how many vectors were indexed
curl http://localhost:6333/collections/mnm_storage | jq '.result.points_count'
```

Should return a number > 0.

**Once you see "all files indexed", the system is ready to query!**

---

## Common Operations

### Start Services (Normal Use)

```bash
# Start services (use existing images)
docker compose -f docker-compose-mcp.yml up

# Start in background (daemon mode)
docker compose -f docker-compose-mcp.yml up -d
```

**Note:** Subsequent starts are fast since data is already indexed.

### Stop Services

```bash
# Stop services (keeps your data)
docker compose -f docker-compose-mcp.yml down

# Stop and remove ALL data (fresh start)
docker compose -f docker-compose-mcp.yml down -v
```

### After Code Changes

```bash
# Rebuild and restart
docker compose -f docker-compose-mcp.yml up --build

# Rebuild specific service
docker compose -f docker-compose-mcp.yml build indexer
docker compose -f docker-compose-mcp.yml up
```

### Restart Single Service

```bash
# Restart just the indexer (after adding new docs)
docker compose -f docker-compose-mcp.yml restart indexer

# View logs for specific service
docker compose -f docker-compose-mcp.yml logs -f indexer
```

---

## Testing Queries

### Via HTTP API

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is chunking?"}'
```

### Via Browser

- **Qdrant Web UI:** http://localhost:6333/dashboard
- **API Docs:** http://localhost:8001/docs

---

## Managing Documents

### Add New Documents

```bash
# Copy files to your docs directory
cp new_doc.pdf /path/to/your/test_docs/

# Wait 20 minutes (auto-indexing) OR restart indexer immediately:
docker compose -f docker-compose-mcp.yml restart indexer
```

### Check Indexing Status

```bash
# Watch indexer logs
docker compose -f docker-compose-mcp.yml logs -f indexer

# Check vector count
curl http://localhost:6333/collections/mnm_storage
```

---

## Troubleshooting

### View Logs

```bash
# All services
docker compose -f docker-compose-mcp.yml logs

# Follow logs (live)
docker compose -f docker-compose-mcp.yml logs -f

# Specific service
docker compose -f docker-compose-mcp.yml logs indexer
docker compose -f docker-compose-mcp.yml logs qdrant
```

### Check Running Services

```bash
docker compose -f docker-compose-mcp.yml ps
```

### Port Already in Use

```bash
# Check what's using the ports
sudo lsof -i :6333
sudo lsof -i :8001

# Kill the process or change ports in docker-compose-mcp.yml
```

### Complete Reset

```bash
# Stop everything
docker compose -f docker-compose-mcp.yml down -v

# Remove persistent data
rm -rf qdrant_data indexer_data

# Start fresh
docker compose -f docker-compose-mcp.yml up --build
```

---

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Document location (absolute path to your documents)
LOCAL_FILES_PATH=/path/to/your/test_docs

# Embedding model
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
EMBEDDING_SIZE=768
```

After changing `.env`:
```bash
docker compose -f docker-compose-mcp.yml down
docker compose -f docker-compose-mcp.yml up --build
```

---

## Data Locations

```
minima/
├── qdrant_data/          # Vector database (persists across restarts)
├── indexer_data/         # SQLite tracking DB
├── .env                  # Your configuration
└── docker-compose-mcp.yml
```

**Tip:** Backup these directories before major changes!

---

## Quick Checks

```bash
# Is everything running?
docker compose -f docker-compose-mcp.yml ps

# How many vectors indexed?
curl http://localhost:6333/collections/mnm_storage | jq '.result.points_count'

# Test query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Check GPU usage (if using GPU)
nvidia-smi
```

---

## When to Rebuild

| Scenario | Command |
|----------|---------|
| **Normal restart** | `docker compose -f docker-compose-mcp.yml up` |
| **Changed Python code** | `docker compose -f docker-compose-mcp.yml up --build` |
| **Changed requirements.txt** | `docker compose -f docker-compose-mcp.yml up --build` |
| **Changed .env only** | `docker compose -f docker-compose-mcp.yml down && up` |
| **Added new docs** | Just wait 20 min OR `restart indexer` |

---

## Performance Tips

### GPU Acceleration

Check if indexer is using GPU:
```bash
docker compose -f docker-compose-mcp.yml logs indexer | grep cuda
```

Should see: `Loading embedding model: all-mpnet-base-v2 on cuda`

### Cache Reuse

The HuggingFace model cache is mounted from host:
```yaml
volumes:
  - ../.cache:/root/.cache
```

This avoids re-downloading the 400MB embedding model on rebuilds.

---

## MCP Server Setup (Claude Code Integration)

### Native Claude Code (Desktop)

Add to your `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "minima": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/minima/mcp-server",
        "run",
        "minima"
      ]
    }
  }
}
```

Replace `/path/to/minima/mcp-server` with your actual path.

### Containerized Claude Code

**Required Changes:**

**1. Docker run command - add host networking:**
```bash
docker run \
  --add-host=host.docker.internal:host-gateway \
  ...
```

This allows container to reach host's `localhost:8001`.

**2. MCP configuration - set environment variable:**
```json
{
  "mcpServers": {
    "minima": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/minima/mcp-server",
        "run",
        "minima"
      ],
      "env": {
        "MINIMA_INDEXER_HOST": "host.docker.internal"
      }
    }
  }
}
```

**3. Install dependencies in container:**
```bash
cd /path/to/minima/mcp-server
uv sync
```

**Note:** The `requestor.py` already supports `MINIMA_INDEXER_HOST` env var:
```python
INDEXER_HOST = os.getenv("MINIMA_INDEXER_HOST", "localhost")
REQUEST_DATA_URL = f"http://{INDEXER_HOST}:8001/query"
```

---

## See Also

- **Full Guide:** `MINIMA_QDRANT_GUIDE.md` (comprehensive documentation)
- **Container Setup:** `../CONTAINER_MINIMA_SETUP.md` (detailed container integration)
- **Qdrant Docs:** https://qdrant.tech/documentation/
- **Original Minima:** https://github.com/dmayboroda/minima
