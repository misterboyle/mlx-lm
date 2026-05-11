#!/bin/bash
# safe_server.sh - Safely manage mlx_lm.server lifecycle
# Prevents OOM crashes by killing existing servers before starting new ones

set -e

SERVER_PORT=${1:-8080}
MODEL_PATH=${2:-"/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"}
EXTRA_ARGS=${@:3}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== MLX LM Server Manager ===${NC}"

# Step 1: Check if server is already running
echo -e "${YELLOW}Checking for existing server...${NC}"
EXISTING_PID=$(lsof -ti :${SERVER_PORT} 2>/dev/null || true)

if [ -n "$EXISTING_PID" ]; then
    echo -e "${RED}WARNING: Server already running on port ${SERVER_PORT} (PID: ${EXISTING_PID})${NC}"
    echo -e "${YELLOW}Killing existing server to prevent OOM...${NC}"
    kill -9 ${EXISTING_PID} 2>/dev/null || true
    sleep 2
    
    # Verify it's dead
    if lsof -ti :${SERVER_PORT} >/dev/null 2>&1; then
        echo -e "${RED}ERROR: Failed to kill existing server${NC}"
        exit 1
    fi
    echo -e "${GREEN}Existing server killed successfully${NC}"
else
    echo -e "${GREEN}No existing server found${NC}"
fi

# Step 2: Check available memory
echo -e "${YELLOW}Checking system memory...${NC}"
MEM_INFO=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$((MEM_INFO / 1024 / 1024 / 1024))
echo -e "Total RAM: ${MEM_GB}GB"

# Check memory pressure (0 = low, 1 = moderate, 2 = high, 3 = critical)
MEMORY_PRESSURE=$(sysctl -n vm.memory_pressure 2>/dev/null || echo "0")
if [ "$MEMORY_PRESSURE" -gt 1 ]; then
    echo -e "${RED}WARNING: High memory pressure (${MEMORY_PRESSURE})${NC}"
    echo -e "${YELLOW}Consider freeing memory before starting server${NC}"
fi

# Step 3: Start server
echo -e "${YELLOW}Starting server on port ${SERVER_PORT}...${NC}"
echo -e "Model: ${MODEL_PATH}"

cd ~/mlx-lm-turbo
source venv/bin/activate

# Start server in background with nohup
nohup python -m mlx_lm.server \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${SERVER_PORT} \
    ${EXTRA_ARGS} \
    > /tmp/mlx_lm_server.log 2>&1 &

SERVER_PID=$!
echo -e "${GREEN}Server started (PID: ${SERVER_PID})${NC}"

# Step 4: Wait for server to be ready
echo -e "${YELLOW}Waiting for server to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:${SERVER_PORT}/health >/dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        echo -e "Port: ${SERVER_PORT}"
        echo -e "PID: ${SERVER_PID}"
        echo -e "Logs: tail -f /tmp/mlx_lm_server.log"
        exit 0
    fi
    sleep 1
done

echo -e "${RED}ERROR: Server failed to start within 30 seconds${NC}"
echo -e "Check logs: cat /tmp/mlx_lm_server.log"
exit 1
