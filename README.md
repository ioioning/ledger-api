# IPLD-like API Server

Production-ready FastAPI server that validates JSON payloads against schemas and persists them as content-addressed blocks in a local DAG (Directed Acyclic Graph), similar to a blockchain.

## Features

✅ **JSON Schema Validation** - Validate all incoming payloads against predefined schemas  
✅ **Content-Addressed Storage** - Immutable blocks identified by deterministic CIDs  
✅ **Chain Structure** - Each block references the previous block forming a verifiable chain  
✅ **Production Ready** - Security hardening, error handling, logging, monitoring  
✅ **Docker Support** - Containerized deployment with Docker Compose  
✅ **Comprehensive Tests** - Full test coverage with pytest  
✅ **RESTful API** - Clean, documented API with OpenAPI/Swagger  

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server:app --reload --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

## Project Structure

```
.
├── server.py                 # Main FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Docker Compose setup
├── test_server.py           # Comprehensive test suite
├── README.md                # This file
├── schemas/                 # JSON Schema files
│   └── create_user.json    # Example schema
├── ipld_store/             # Block storage (auto-created)
└── ipld_index.json         # Chain head index (auto-created)
```

## API Endpoints

### Health & Metadata

- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /schemas` - List available schemas
- `GET /chains` - List all chains

### Chain Operations

- `POST /chains/{chain}/append/{schema}` - Append validated payload to chain
- `GET /chains/{chain}/head` - Get current head CID
- `GET /chains/{chain}/blocks` - List blocks in chain
- `GET /chains/{chain}/blocks/{cid}` - Get specific block by CID

## Usage Examples

### 1. Create a Schema

Create `schemas/create_user.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "username": {"type": "string", "minLength": 3},
    "email": {"type": "string", "format": "email"}
  },
  "required": ["username", "email"],
  "additionalProperties": false
}
```

### 2. Append Data to Chain

```bash
curl -X POST "http://localhost:8000/chains/users/append/create_user" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com"
  }'
```

Response:
```json
{
  "cid": "cid:a1b2c3d4...",
  "chain": "users",
  "timestamp": 1699564800.123
}
```

### 3. Get Chain Head

```bash
curl "http://localhost:8000/chains/users/head"
```

Response:
```json
{
  "chain": "users",
  "head": "cid:a1b2c3d4..."
}
```

### 4. List Blocks

```bash
curl "http://localhost:8000/chains/users/blocks?limit=10"
```

Response:
```json
{
  "chain": "users",
  "head": "cid:a1b2c3d4...",
  "blocks": [
    {
      "cid": "cid:a1b2c3d4...",
      "timestamp": 1699564800.123,
      "previous": "cid:x9y8z7...",
      "has_payload": true
    }
  ],
  "count": 10
}
```

### 5. Get Specific Block

```bash
curl "http://localhost:8000/chains/users/blocks/cid:a1b2c3d4..."
```

Response:
```json
{
  "cid": "cid:a1b2c3d4...",
  "block": {
    "payload": {
      "username": "alice",
      "email": "alice@example.com"
    },
    "timestamp": 1699564800.123,
    "previous": "cid:x9y8z7...",
    "meta": {
      "schema": "create_user"
    },
    "chain": "users"
  }
}
```

## Configuration

Configure via environment variables:

```bash
# Storage paths
SCHEMA_DIR=./schemas
IPLD_DIR=./ipld_store
INDEX_FILE=./ipld_index.json

# Security
MAX_PAYLOAD_SIZE=1048576         # 1MB
MAX_CHAIN_NAME_LENGTH=100
MAX_SCHEMA_NAME_LENGTH=100
ALLOWED_HOSTS=*
CORS_ORIGINS=*

# Performance
MAX_TRAVERSE_STEPS=10000
DEFAULT_LIST_LIMIT=50
MAX_LIST_LIMIT=1000
```

## Testing

```bash
# Run all tests
pytest test_server.py -v

# Run with coverage
pytest test_server.py -v --cov=server --cov-report=html

# Run specific test
pytest test_server.py -v -k test_append_to_chain_valid
```

## Security Features

- ✅ Input validation (chain names, schema names, CIDs)
- ✅ Payload size limits (prevents DoS)
- ✅ Non-root container user
- ✅ Path traversal prevention
- ✅ CORS configuration
- ✅ Trusted host middleware
- ✅ Comprehensive error handling
- ✅ Atomic file operations

## Architecture

### Block Structure

Each block contains:

```json
{
  "payload": {},          // Your validated data
  "timestamp": 1234.56,   // Unix timestamp
  "previous": "cid:...",  // Previous block CID (null for genesis)
  "meta": {},             // Optional metadata
  "chain": "chain_name"   // Chain identifier
}
```

### CID Computation

CIDs are computed deterministically:

1. Convert block to canonical JSON (sorted keys, no whitespace)
2. Compute SHA-256 hash
3. Prefix with `"cid:"` to create content identifier

### Storage

- **Blocks**: Stored as individual JSON files in `ipld_store/`
- **Index**: `ipld_index.json` maps chain names to head CIDs
- **Atomic**: All writes use atomic file operations (write to temp, then rename)

## Production Considerations

### Scaling

For high-traffic production use:

1. **Database Backend**: Replace file storage with PostgreSQL/MongoDB
2. **Caching**: Add Redis for frequently accessed blocks
3. **Load Balancing**: Deploy multiple instances behind load balancer
4. **Object Storage**: Use S3/GCS for block storage
5. **CDN**: Cache static content and block retrievals

### Monitoring

Add monitoring with Prometheus:

```python
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

### Authentication

Add API key or JWT authentication:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/chains/{chain}/append/{schema}")
async def append_to_chain(
    chain: str,
    schema: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate credentials
    ...
```

### Rate Limiting

Add rate limiting middleware:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chains/{chain}/append/{schema}")
@limiter.limit("10/minute")
async def append_to_chain(...):
    ...
```

## Future Enhancements

- [ ] Block signing with cryptographic signatures
- [ ] IPFS integration for distributed storage
- [ ] GraphQL API alongside REST
- [ ] Webhook notifications for new blocks
- [ ] Block pruning and archival
- [ ] Multi-tenant support
- [ ] Schema versioning and migration
- [ ] Full-text search across payloads
- [ ] Merkle tree proofs for block verification

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review API docs at `/docs` endpoint