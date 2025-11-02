# Minima Quick Start

Quick reference for running your local RAG system with Qdrant + Minima.

---

## First Time Setup

```bash
cd /home/lostica/projects/mcp_rag/minima

# Build and start everything (takes ~5 min first time)
docker compose -f docker-compose-mcp.yml up --build
```

**Wait for:** `"all files indexed"` message in logs.

---

## Common Operations

### Start Services (Normal Use)

```bash
# Start services (use existing images)
docker compose -f docker-compose-mcp.yml up

# Start in background (daemon mode)
docker compose -f docker-compose-mcp.yml up -d
```

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
cp new_doc.pdf /home/lostica/projects/mcp_rag/test_docs/

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
# Document location
LOCAL_FILES_PATH=/home/lostica/projects/mcp_rag/test_docs

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

## See Also

- **Full Guide:** `MINIMA_QDRANT_GUIDE.md` (comprehensive documentation)
- **Qdrant Docs:** https://qdrant.tech/documentation/
- **Original Minima:** https://github.com/dmayboroda/minima
