"""
Production-ready IPLD-like API server with JSON Schema validation.
Validates incoming JSON payloads and persists them as content-addressed blocks.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from jsonschema import validate, ValidationError
from typing import Optional, Dict, Any, List
import os
import json
import hashlib
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Application configuration with environment variable support."""
    
    SCHEMA_DIR = Path(os.getenv("SCHEMA_DIR", "./schemas"))
    IPLD_DIR = Path(os.getenv("IPLD_DIR", "./ipld_store"))
    INDEX_FILE = Path(os.getenv("INDEX_FILE", "./ipld_index.json"))
    
    # Security settings
    MAX_PAYLOAD_SIZE = int(os.getenv("MAX_PAYLOAD_SIZE", 1024 * 1024))  # 1MB default
    MAX_CHAIN_NAME_LENGTH = int(os.getenv("MAX_CHAIN_NAME_LENGTH", 100))
    MAX_SCHEMA_NAME_LENGTH = int(os.getenv("MAX_SCHEMA_NAME_LENGTH", 100))
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Performance settings
    MAX_TRAVERSE_STEPS = int(os.getenv("MAX_TRAVERSE_STEPS", 10000))
    DEFAULT_LIST_LIMIT = int(os.getenv("DEFAULT_LIST_LIMIT", 50))
    MAX_LIST_LIMIT = int(os.getenv("MAX_LIST_LIMIT", 1000))

config = Config()

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

def initialize_storage():
    """Initialize storage directories and index file."""
    try:
        config.SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        config.IPLD_DIR.mkdir(parents=True, exist_ok=True)
        
        if not config.INDEX_FILE.exists():
            config.INDEX_FILE.write_text(json.dumps({}, indent=2))
            logger.info("Initialized index file")
        
        logger.info(f"Storage initialized: {config.IPLD_DIR}")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting IPLD API server...")
    initialize_storage()
    yield
    logger.info("Shutting down IPLD API server...")

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IPLD-like API Server",
    description="Content-addressed block storage with JSON Schema validation",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware
if config.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request size limiter middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    """Limit request payload size to prevent DoS attacks."""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.MAX_PAYLOAD_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": f"Payload too large. Max size: {config.MAX_PAYLOAD_SIZE} bytes"}
            )
    return await call_next(request)

# ---------------------------------------------------------------------------
# Utilities: canonical JSON, CID generation, persistence
# ---------------------------------------------------------------------------

def canonical_json_bytes(obj: Any) -> bytes:
    """Return a canonical byte representation of JSON object."""
    return json.dumps(obj, separators=(',', ':'), sort_keys=True).encode('utf-8')

def compute_cid(obj: Any) -> str:
    """
    Compute a CID (content identifier) for an object.
    Uses SHA-256 hash of canonical JSON representation.
    """
    digest = hashlib.sha256(canonical_json_bytes(obj)).hexdigest()
    return f"cid:{digest}"

def read_index() -> Dict[str, str]:
    """Read the chain index file with error handling."""
    try:
        return json.loads(config.INDEX_FILE.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Index file corrupted: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to read index: {e}")
        raise HTTPException(status_code=500, detail="Internal storage error")

def write_index(index: Dict[str, str]):
    """Write chain index with atomic operation."""
    try:
        # Write to temporary file first
        temp_file = config.INDEX_FILE.with_suffix('.tmp')
        temp_file.write_text(json.dumps(index, indent=2))
        # Atomic rename
        temp_file.replace(config.INDEX_FILE)
    except Exception as e:
        logger.error(f"Failed to write index: {e}")
        raise HTTPException(status_code=500, detail="Internal storage error")

def persist_block(chain: str, block_obj: Dict[str, Any]) -> str:
    """
    Persist a block object to disk and update chain head.
    Returns the computed CID.
    """
    try:
        # Compute CID deterministically
        cid = compute_cid(block_obj)
        
        # Write block file atomically
        filename = config.IPLD_DIR / f"{cid}.json"
        temp_filename = filename.with_suffix('.tmp')
        temp_filename.write_text(json.dumps(block_obj, indent=2, sort_keys=True))
        temp_filename.replace(filename)
        
        # Update index (chain head)
        index = read_index()
        index[chain] = cid
        write_index(index)
        
        logger.info(f"Persisted block {cid} to chain {chain}")
        return cid
    except Exception as e:
        logger.error(f"Failed to persist block: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist block")

def load_block(cid: str) -> Dict[str, Any]:
    """Load a block from disk by CID."""
    filename = config.IPLD_DIR / f"{cid}.json"
    if not filename.exists():
        raise HTTPException(status_code=404, detail=f"Block {cid} not found")
    try:
        return json.loads(filename.read_text())
    except json.JSONDecodeError:
        logger.error(f"Corrupted block file: {cid}")
        raise HTTPException(status_code=500, detail="Corrupted block data")

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_chain_name(chain: str):
    """Validate chain name to prevent path traversal and injection."""
    if not chain or len(chain) > config.MAX_CHAIN_NAME_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid chain name length (max {config.MAX_CHAIN_NAME_LENGTH})"
        )
    if not chain.replace('_', '').replace('-', '').isalnum():
        raise HTTPException(
            status_code=400,
            detail="Chain name must be alphanumeric (underscore and dash allowed)"
        )

def validate_schema_name(schema_name: str):
    """Validate schema name to prevent path traversal."""
    if not schema_name or len(schema_name) > config.MAX_SCHEMA_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema name length (max {config.MAX_SCHEMA_NAME_LENGTH})"
        )
    if not schema_name.replace('_', '').isalnum():
        raise HTTPException(
            status_code=400,
            detail="Schema name must be alphanumeric (underscore allowed)"
        )

def validate_cid(cid: str):
    """Validate CID format."""
    if not cid.startswith("cid:") or len(cid) != 68:  # "cid:" + 64 hex chars
        raise HTTPException(status_code=400, detail="Invalid CID format")

# ---------------------------------------------------------------------------
# Schema loader / validator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """Loads and validates JSON schemas with caching."""
    
    def __init__(self, schema_dir: Path = config.SCHEMA_DIR):
        self.schema_dir = schema_dir
        self._cache: Dict[str, Dict] = {}
    
    def _load_schema(self, name: str) -> Dict:
        """Load schema from disk with caching."""
        if name in self._cache:
            return self._cache[name]
        
        path = self.schema_dir / f"{name}.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Schema '{name}' not found"
            )
        
        try:
            schema = json.loads(path.read_text())
            self._cache[name] = schema
            logger.info(f"Loaded schema: {name}")
            return schema
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema {name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Schema '{name}' contains invalid JSON"
            )
    
    def validate(self, schema_name: str, obj: Any):
        """Validate object against schema."""
        schema = self._load_schema(schema_name)
        validate(instance=obj, schema=schema)

schema_validator = SchemaValidator()

# ---------------------------------------------------------------------------
# IPLD Chain Manager
# ---------------------------------------------------------------------------

class IPLDChain:
    """Manager for chain operations: append and traverse."""
    
    def __init__(self, chain_name: str):
        validate_chain_name(chain_name)
        self.chain = chain_name
    
    def head(self) -> Optional[str]:
        """Get the current head CID of the chain."""
        index = read_index()
        return index.get(self.chain)
    
    def append(self, payload: Any, meta: Optional[Dict] = None) -> str:
        """Create and persist a new block on the chain."""
        previous = self.head()
        block_obj = {
            "payload": payload,
            "timestamp": time.time(),
            "previous": previous,
            "meta": meta or {},
            "chain": self.chain
        }
        cid = persist_block(self.chain, block_obj)
        return cid
    
    def get(self, cid: str) -> Dict[str, Any]:
        """Get a block by CID."""
        validate_cid(cid)
        return load_block(cid)
    
    def traverse_from(
        self, 
        cid: Optional[str] = None, 
        max_steps: int = config.MAX_TRAVERSE_STEPS
    ):
        """
        Yield blocks starting from CID (or head if None) walking backwards.
        """
        current = cid or self.head()
        steps = 0
        
        while current and steps < max_steps:
            try:
                block = self.get(current)
                yield current, block
                current = block.get('previous')
                steps += 1
            except HTTPException:
                break

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "IPLD API Server",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "storage": {
            "schemas": config.SCHEMA_DIR.exists(),
            "blocks": config.IPLD_DIR.exists(),
            "index": config.INDEX_FILE.exists()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post(
    "/chains/{chain}/append/{schema_name}",
    status_code=status.HTTP_201_CREATED,
    tags=["chains"]
)
async def append_to_chain(chain: str, schema_name: str, request: Request):
    """
    Append a validated JSON payload to a named chain.
    
    - **chain**: Name of the chain (alphanumeric, underscore, dash)
    - **schema_name**: Name of schema file (without .json extension)
    
    The request body must be valid JSON matching the schema.
    Returns the new block's CID on success.
    """
    validate_chain_name(chain)
    validate_schema_name(schema_name)
    
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"Invalid JSON in request: {e}")
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON"
        )
    
    # Validate against schema
    try:
        schema_validator.validate(schema_name, payload)
    except ValidationError as e:
        logger.info(f"Schema validation failed for {schema_name}: {e.message}")
        raise HTTPException(
            status_code=422,
            detail={
                "message": e.message,
                "path": list(e.path),
                "schema_path": list(e.schema_path)
            }
        )
    
    # Append to chain
    chain_mgr = IPLDChain(chain)
    cid = chain_mgr.append(payload, meta={"schema": schema_name})
    
    return {
        "cid": cid,
        "chain": chain,
        "timestamp": time.time()
    }

@app.get("/chains/{chain}/head", tags=["chains"])
async def get_head(chain: str):
    """Get the current head CID for a chain."""
    validate_chain_name(chain)
    chain_mgr = IPLDChain(chain)
    head = chain_mgr.head()
    
    if not head:
        raise HTTPException(
            status_code=404,
            detail=f"Chain '{chain}' not found or empty"
        )
    
    return {"chain": chain, "head": head}

@app.get("/chains/{chain}/blocks/{cid}", tags=["chains"])
async def get_block(chain: str, cid: str):
    """
    Fetch a specific block by CID.
    Verifies the block belongs to the chain.
    """
    validate_chain_name(chain)
    validate_cid(cid)
    
    chain_mgr = IPLDChain(chain)
    head = chain_mgr.head()
    
    if not head:
        raise HTTPException(
            status_code=404,
            detail=f"Chain '{chain}' not found or empty"
        )
    
    # Traverse chain to verify block belongs to it
    for c, block in chain_mgr.traverse_from(head):
        if c == cid:
            return {"cid": c, "block": block}
    
    raise HTTPException(
        status_code=404,
        detail=f"Block '{cid}' not found in chain '{chain}'"
    )

@app.get("/chains/{chain}/blocks", tags=["chains"])
async def list_chain(chain: str, limit: int = config.DEFAULT_LIST_LIMIT):
    """
    List blocks in a chain from head backwards.
    
    - **limit**: Maximum number of blocks to return (default: 50, max: 1000)
    """
    validate_chain_name(chain)
    
    if limit < 1 or limit > config.MAX_LIST_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Limit must be between 1 and {config.MAX_LIST_LIMIT}"
        )
    
    chain_mgr = IPLDChain(chain)
    head = chain_mgr.head()
    
    if not head:
        raise HTTPException(
            status_code=404,
            detail=f"Chain '{chain}' not found or empty"
        )
    
    blocks = []
    for cid, block in chain_mgr.traverse_from(head, max_steps=limit):
        blocks.append({
            "cid": cid,
            "timestamp": block.get('timestamp'),
            "previous": block.get('previous'),
            "has_payload": 'payload' in block
        })
        if len(blocks) >= limit:
            break
    
    return {
        "chain": chain,
        "head": head,
        "blocks": blocks,
        "count": len(blocks)
    }

@app.get("/chains", tags=["chains"])
async def list_chains():
    """List all available chains."""
    index = read_index()
    chains = [
        {"name": name, "head": head}
        for name, head in index.items()
    ]
    return {"chains": chains, "count": len(chains)}

@app.get("/schemas", tags=["schemas"])
async def list_schemas():
    """List all available schemas."""
    try:
        schemas = [
            f.stem for f in config.SCHEMA_DIR.glob("*.json")
        ]
        return {"schemas": schemas, "count": len(schemas)}
    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        raise HTTPException(status_code=500, detail="Failed to list schemas")

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle JSON schema validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "message": exc.message,
                "path": list(exc.path),
                "schema_path": list(exc.schema_path)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )