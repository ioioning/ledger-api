"""
Comprehensive test suite for IPLD API server.
Run with: pytest test_server.py -v --cov=server
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from server import app, config, initialize_storage, IPLDChain, compute_cid

# Test fixtures
@pytest.fixture(scope="function")
def temp_storage(tmp_path):
    """Create temporary storage for each test."""
    # Override config paths
    config.SCHEMA_DIR = tmp_path / "schemas"
    config.IPLD_DIR = tmp_path / "ipld_store"
    config.INDEX_FILE = tmp_path / "ipld_index.json"
    
    # Initialize storage
    initialize_storage()
    
    # Create a test schema
    config.SCHEMA_DIR.mkdir(exist_ok=True)
    test_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "username": {"type": "string", "minLength": 3},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["username", "email"],
        "additionalProperties": False
    }
    (config.SCHEMA_DIR / "test_user.json").write_text(json.dumps(test_schema))
    
    yield tmp_path
    
    # Cleanup is automatic with tmp_path

@pytest.fixture
def client(temp_storage):
    """Create test client."""
    return TestClient(app)

# Health check tests
def test_root_endpoint(client):
    """Test root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_health_endpoint(client):
    """Test detailed health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "storage" in data
    assert data["storage"]["schemas"] is True

# Schema listing tests
def test_list_schemas(client):
    """Test schema listing endpoint."""
    response = client.get("/schemas")
    assert response.status_code == 200
    data = response.json()
    assert "schemas" in data
    assert "test_user" in data["schemas"]

# Chain tests
def test_list_chains_empty(client):
    """Test listing chains when none exist."""
    response = client.get("/chains")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["chains"] == []

def test_append_to_chain_valid(client):
    """Test appending valid payload to chain."""
    payload = {
        "username": "alice",
        "email": "alice@example.com"
    }
    response = client.post(
        "/chains/users/append/test_user",
        json=payload
    )
    assert response.status_code == 201
    data = response.json()
    assert "cid" in data
    assert data["cid"].startswith("cid:")
    assert data["chain"] == "users"

def test_append_to_chain_invalid_payload(client):
    """Test appending invalid payload."""
    payload = {
        "username": "ab",  # Too short
        "email": "invalid-email"
    }
    response = client.post(
        "/chains/users/append/test_user",
        json=payload
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_append_to_chain_missing_schema(client):
    """Test appending with non-existent schema."""
    payload = {"test": "data"}
    response = client.post(
        "/chains/test/append/nonexistent",
        json=payload
    )
    assert response.status_code == 404

def test_append_multiple_blocks(client):
    """Test appending multiple blocks to form a chain."""
    payloads = [
        {"username": "alice", "email": "alice@example.com"},
        {"username": "bob", "email": "bob@example.com"},
        {"username": "charlie", "email": "charlie@example.com"}
    ]
    
    cids = []
    for payload in payloads:
        response = client.post(
            "/chains/users/append/test_user",
            json=payload
        )
        assert response.status_code == 201
        cids.append(response.json()["cid"])
    
    # Verify chain structure
    response = client.get("/chains/users/blocks")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert data["head"] == cids[-1]  # Last CID is head

def test_get_chain_head(client):
    """Test getting chain head."""
    # First append a block
    payload = {"username": "alice", "email": "alice@example.com"}
    response = client.post(
        "/chains/users/append/test_user",
        json=payload
    )
    cid = response.json()["cid"]
    
    # Get head
    response = client.get("/chains/users/head")
    assert response.status_code == 200
    data = response.json()
    assert data["head"] == cid

def test_get_chain_head_nonexistent(client):
    """Test getting head of non-existent chain."""
    response = client.get("/chains/nonexistent/head")
    assert response.status_code == 404

def test_get_block_by_cid(client):
    """Test retrieving a specific block."""
    payload = {"username": "alice", "email": "alice@example.com"}
    response = client.post(
        "/chains/users/append/test_user",
        json=payload
    )
    cid = response.json()["cid"]
    
    # Get block
    response = client.get(f"/chains/users/blocks/{cid}")
    assert response.status_code == 200
    data = response.json()
    assert data["cid"] == cid
    assert data["block"]["payload"] == payload

def test_get_block_invalid_cid(client):
    """Test getting block with invalid CID format."""
    response = client.get("/chains/users/blocks/invalid-cid")
    assert response.status_code == 400

def test_list_blocks_with_limit(client):
    """Test listing blocks with custom limit."""
    # Add 5 blocks
    for i in range(5):
        payload = {
            "username": f"user{i}",
            "email": f"user{i}@example.com"
        }
        client.post("/chains/users/append/test_user", json=payload)
    
    # List with limit
    response = client.get("/chains/users/blocks?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3

def test_list_blocks_invalid_limit(client):
    """Test listing blocks with invalid limit."""
    response = client.get("/chains/users/blocks?limit=10000")
    assert response.status_code == 400

# Input validation tests
def test_invalid_chain_name(client):
    """Test invalid chain names."""
    invalid_names = [
        "../etc",
        "chain/../path",
        "chain with spaces",
        "chain@special",
        "a" * 200  # Too long
    ]
    
    payload = {"username": "alice", "email": "alice@example.com"}
    for name in invalid_names:
        response = client.post(
            f"/chains/{name}/append/test_user",
            json=payload
        )
        assert response.status_code == 400

def test_invalid_schema_name(client):
    """Test invalid schema names."""
    invalid_names = [
        "../etc/passwd",
        "schema with spaces",
        "schema@special"
    ]
    
    payload = {"test": "data"}
    for name in invalid_names:
        response = client.post(
            f"/chains/test/append/{name}",
            json=payload
        )
        assert response.status_code == 400

def test_payload_size_limit(client):
    """Test payload size limiting."""
    # Create large payload
    large_payload = {
        "username": "alice",
        "email": "alice@example.com",
        "data": "x" * (config.MAX_PAYLOAD_SIZE + 1000)
    }
    
    response = client.post(
        "/chains/users/append/test_user",
        json=large_payload
    )
    assert response.status_code == 413

def test_malformed_json(client):
    """Test handling of malformed JSON."""
    response = client.post(
        "/chains/users/append/test_user",
        content="{invalid json}",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400

# CID computation tests
def test_cid_determinism():
    """Test that CID computation is deterministic."""
    obj = {"key": "value", "number": 42}
    cid1 = compute_cid(obj)
    cid2 = compute_cid(obj)
    assert cid1 == cid2

def test_cid_different_order():
    """Test that key order doesn't affect CID."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}
    assert compute_cid(obj1) == compute_cid(obj2)

def test_cid_different_content():
    """Test that different content produces different CIDs."""
    obj1 = {"key": "value1"}
    obj2 = {"key": "value2"}
    assert compute_cid(obj1) != compute_cid(obj2)

# Chain traversal tests
def test_chain_traversal(temp_storage):
    """Test chain traversal functionality."""
    chain = IPLDChain("test")
    
    # Add blocks
    cids = []
    for i in range(5):
        cid = chain.append({"index": i})
        cids.append(cid)
    
    # Traverse chain
    traversed = list(chain.traverse_from())
    assert len(traversed) == 5
    
    # Verify order (newest to oldest)
    for i, (cid, block) in enumerate(traversed):
        assert block["payload"]["index"] == 4 - i

def test_chain_traversal_with_limit(temp_storage):
    """Test chain traversal with step limit."""
    chain = IPLDChain("test")
    
    # Add 10 blocks
    for i in range(10):
        chain.append({"index": i})
    
    # Traverse with limit
    traversed = list(chain.traverse_from(max_steps=5))
    assert len(traversed) == 5

# Integration tests
def test_full_workflow(client):
    """Test complete workflow: create chain, add blocks, retrieve."""
    # 1. Create chain by adding first block
    payload1 = {"username": "alice", "email": "alice@example.com"}
    response = client.post("/chains/users/append/test_user", json=payload1)
    assert response.status_code == 201
    cid1 = response.json()["cid"]
    
    # 2. Add second block
    payload2 = {"username": "bob", "email": "bob@example.com"}
    response = client.post("/chains/users/append/test_user", json=payload2)
    assert response.status_code == 201
    cid2 = response.json()["cid"]
    
    # 3. Get chain head
    response = client.get("/chains/users/head")
    assert response.status_code == 200
    assert response.json()["head"] == cid2
    
    # 4. List blocks
    response = client.get("/chains/users/blocks")
    assert response.status_code == 200
    blocks = response.json()["blocks"]
    assert len(blocks) == 2
    assert blocks[0]["cid"] == cid2  # Most recent first
    assert blocks[1]["cid"] == cid1
    
    # 5. Get specific block
    response = client.get(f"/chains/users/blocks/{cid1}")
    assert response.status_code == 200
    assert response.json()["block"]["payload"] == payload1
    
    # 6. List all chains
    response = client.get("/chains")
    assert response.status_code == 200
    chains = response.json()["chains"]
    assert len(chains) == 1
    assert chains[0]["name"] == "users"

# Error handling tests
def test_corrupted_index_file(client, temp_storage):
    """Test handling of corrupted index file."""
    # Corrupt the index
    config.INDEX_FILE.write_text("invalid json{")
    
    # Should handle gracefully
    response = client.get("/chains")
    assert response.status_code == 200

# Performance tests
def test_large_chain_performance(client):
    """Test performance with larger chain."""
    import time
    
    # Add 100 blocks
    start = time.time()
    for i in range(100):
        payload = {
            "username": f"user{i}",
            "email": f"user{i}@example.com"
        }
        response = client.post("/chains/perf/append/test_user", json=payload)
        assert response.status_code == 201
    duration = time.time() - start
    
    # Should complete in reasonable time
    assert duration < 10  # 10 seconds for 100 blocks
    
    # Verify chain
    response = client.get("/chains/perf/blocks?limit=100")
    assert response.status_code == 200
    assert response.json()["count"] == 100