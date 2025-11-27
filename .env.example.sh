# IPLD API Server Configuration
# Copy this file to .env and adjust values as needed

# Storage Configuration
SCHEMA_DIR=./schemas
IPLD_DIR=./ipld_store
INDEX_FILE=./ipld_index.json

# Security Settings
MAX_PAYLOAD_SIZE=1048576              # Maximum payload size in bytes (1MB)
MAX_CHAIN_NAME_LENGTH=100             # Maximum chain name length
MAX_SCHEMA_NAME_LENGTH=100            # Maximum schema name length
ALLOWED_HOSTS=*                       # Comma-separated list of allowed hosts (* for all)
CORS_ORIGINS=*                        # Comma-separated list of CORS origins (* for all)

# Performance Settings
MAX_TRAVERSE_STEPS=10000              # Maximum steps when traversing chain
DEFAULT_LIST_LIMIT=50                 # Default number of blocks to list
MAX_LIST_LIMIT=1000                   # Maximum number of blocks that can be listed

# Production Settings (uncomment for production)
# ALLOWED_HOSTS=api.example.com,www.example.com
# CORS_ORIGINS=https://example.com,https://app.example.com
# MAX_PAYLOAD_SIZE=524288             # Reduce to 512KB for production